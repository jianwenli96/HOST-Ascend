# coding=utf-8
"""Base class for defining training algorithm."""

import abc
import torch
import torch.nn as nn
import zlib
from config import CONFIG
from models import get_model
from utils import get_cnn_feats


class RefEmbeddingCache:
    """LRU cache for ref embeddings (eval only).

    key: tuple(sample_ref_frame_paths) — full M*NUM_STEPS frame paths per sample
    value: emb_ref — (M*NUM_STEPS, D), post-emb, already chunk-merged

    index: optional Manager().dict() shared with DataLoader workers (int key → True).
           Workers query index to determine hit/miss without touching CUDA tensors.
    """
    def __init__(self, maxsize=256, index=None):
        self._store = {}         # key → (emb_ref,)
        self._order = []         # LRU order
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.last_hit = None   # last cache query result (True/False/None)
        self.index = index if index is not None else {}  # shared index for workers

    def _hash_paths(self, ref_paths):
        return zlib.crc32('\0'.join(ref_paths).encode())

    def get(self, ref_paths):
        key = self._hash_paths(ref_paths)
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            self.hits += 1
            self.last_hit = True
            return self._store[key]
        self.misses += 1
        self.last_hit = False
        return None

    def put(self, ref_paths, emb_ref):
        key = self._hash_paths(ref_paths)
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self._maxsize:
            oldest = self._order.pop(0)
            del self._store[oldest]
            self.index.pop(oldest, None)   # evict from shared index too
        self._store[key] = (emb_ref,)
        self._order.append(key)
        self.index[key] = True             # sync to shared index

    def stats(self):
        total = self.hits + self.misses
        return self.hits, self.misses, self.hits / total if total > 0 else 0.0


class Algorithm(nn.Module):
    """Base class for defining algorithms."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, model=None):
        super(Algorithm, self).__init__()
        if model:
            self.model = model
        else:
            self.model = get_model()
            
        if isinstance(self.model, dict):
            self.cnn = self.model['cnn']
            self.emb = self.model['emb']
            if 'gate' in self.model:
                self.gate = self.model['gate']
        else:
            # If it's already a module
            pass

        self._ref_cache = RefEmbeddingCache(
            maxsize=getattr(CONFIG.EVAL, 'REF_CACHE_MAXSIZE', 256))

    def forward(self, data, steps, seq_lens, training=True):
        """One pass through the model."""
        if training:
            num_steps = CONFIG.TRAIN.NUM_FRAMES
            self.train()
            all_hit = False
        else:
            num_steps = CONFIG.EVAL.NUM_FRAMES
            self.eval()
            # shallow copy so we don't mutate the caller's dict
            data = dict(data)
            # Check cache before Qwen forward: all refs hit → main_only; else → paired
            if getattr(CONFIG.EVAL, 'REF_CACHE_MAIN_ONLY', True):
                ref_paths_list = data.get('ref_frame_paths', [])
                if ref_paths_list:
                    all_hit = all(self._ref_cache.get(rp) is not None for rp in ref_paths_list)
                    self._ref_cache.last_hit = all_hit
                else:
                    all_hit = True
                    self._ref_cache.last_hit = True
                if all_hit:
                    data['qwen_input'] = data.get('qwen_input_main_only', data['qwen_input'])
                else:
                    data['qwen_input'] = data.get('qwen_input_paired', data['qwen_input'])
            else:
                all_hit = False
                data['qwen_input'] = data.get('qwen_input_paired', data['qwen_input'])

        cnn_feats = get_cnn_feats(self.cnn, data, training)
        
        raw_tokens_dict = None
        if isinstance(cnn_feats, dict) and 'embeddings' in cnn_feats:
            raw_tokens_dict = {
                'mains_tokens': cnn_feats['mains_tokens'],
                'refs_tokens': cnn_feats['refs_tokens']
            }
            cnn_feats = cnn_feats['embeddings']

        embs = self.emb(cnn_feats, num_steps)
        
        channels = embs.size(-1)
        embs = embs.view(-1, num_steps, channels)
        
        data['qwen_input'] = data.get('qwen_input_paired', data['qwen_input'])

        # ---- Ref embedding cache (eval only, all_hit already decided above) ----
        if all_hit:
            # Cache hit: Qwen ran on main_only only; concat cached refs
            embs_main = embs
            n = len(data['ref_frame_paths']) // 2
            ref_paths = data['ref_frame_paths'][n:]
            embs_ref = torch.stack([
                self._ref_cache.get(rp)[0] for rp in ref_paths
            ])   # (B, T, D)
            embs = torch.cat([embs_main, embs_ref], dim=0)   # (2B, T, D)
            # Restore paired metadata for subsequent merge logic
            data['qwen_input'] = data.get('qwen_input_paired', data['qwen_input'])
        else:
            print("caching ref embeddings")
            # Cache miss: Qwen ran on paired; split and populate cache
            B = embs.size(0) // 2
            embs_main = embs[:B]
            embs_ref  = embs[B:]
            n = len(data['ref_frame_paths']) // 2
            ref_paths = data['ref_frame_paths'][n:]
            for i, rp in enumerate(ref_paths):
                self._ref_cache.put(rp, embs_ref[i])

        # --- Merge Chunks and Call Gate in Forward ---
        # qwen_meta must come from paired data so that group_ids/chunk_ids describe both halves
        qwen_meta = (data.get('qwen_input_paired') or data.get('qwen_input')) \
                    if isinstance(data, dict) else None

        if qwen_meta is not None and 'group_ids' in qwen_meta:
            real_batch_size = embs_main.size(0)   # already split above
            gids = qwen_meta['group_ids'][:real_batch_size]
            cids = qwen_meta['chunk_ids'][:real_batch_size]
            unique_gids = torch.unique(gids)

            steps_main = steps[:real_batch_size]
            steps_ref = steps[real_batch_size:]
            seq_lens_main = seq_lens[:real_batch_size]
            seq_lens_ref = seq_lens[real_batch_size:]
            
            # Masks & Dustbin Flags
            has_cut_masks = 'is_cut_masks' in qwen_meta and qwen_meta['is_cut_masks'] is not None
            cut_masks = qwen_meta['is_cut_masks'] if has_cut_masks else None 
            has_db_flags = 'has_dustbin' in qwen_meta and qwen_meta['has_dustbin'] is not None
            db_flags = qwen_meta['has_dustbin'] if has_db_flags else None
            
            if has_cut_masks:
                 cut_masks_ref = cut_masks[real_batch_size:]
            if has_db_flags:
                 db_flags_ref = db_flags[real_batch_size:]

            new_embs_main, new_steps_main, new_seq_lens_main = [], [], []
            new_embs_ref, new_steps_ref, new_seq_lens_ref = [], [], []
            new_tokens_m, new_tokens_r = [], []
            new_cut_masks_ref, new_db_flags_ref = [], []
            merged_meta_list = []

            # Handle tokens if present
            m_tokens_all, r_tokens_all = None, None
            if raw_tokens_dict is not None:
                m_tokens_all = raw_tokens_dict['mains_tokens'].view(-1, num_steps, raw_tokens_dict['mains_tokens'].shape[-2], raw_tokens_dict['mains_tokens'].shape[-1])
                r_tokens_all = raw_tokens_dict['refs_tokens'].view(-1, num_steps, raw_tokens_dict['refs_tokens'].shape[-2], raw_tokens_dict['refs_tokens'].shape[-1])

            for gid in unique_gids:
                mask = (gids == gid)
                chunk_idx = torch.argsort(cids[mask])

                # Merge Main
                curr_embs_m = embs_main[mask][chunk_idx]
                new_embs_main.append(curr_embs_m.reshape(-1, curr_embs_m.size(-1)))
                curr_steps_m = steps_main[mask][chunk_idx]
                new_steps_main.append(curr_steps_m.reshape(-1))
                new_seq_lens_main.append(seq_lens_main[mask][0])

                # Merge Ref
                curr_embs_r = embs_ref[mask][chunk_idx]
                new_embs_ref.append(curr_embs_r.reshape(-1, curr_embs_r.size(-1)))
                curr_steps_r = steps_ref[mask][chunk_idx]
                new_steps_ref.append(curr_steps_r.reshape(-1))
                new_seq_lens_ref.append(seq_lens_ref[mask][0])

                # Merge Tokens
                if m_tokens_all is not None:
                    curr_tokens_m = m_tokens_all[mask][chunk_idx]
                    new_tokens_m.append(curr_tokens_m.reshape(-1, curr_tokens_m.size(-2), curr_tokens_m.size(-1)))
                    curr_tokens_r = r_tokens_all[mask][chunk_idx]
                    new_tokens_r.append(curr_tokens_r.reshape(-1, curr_tokens_r.size(-2), curr_tokens_r.size(-1)))

                # Merge Masks
                if has_cut_masks:
                    curr_mask_r = cut_masks_ref[mask][chunk_idx]
                    new_cut_masks_ref.append(curr_mask_r.reshape(-1))
                if has_db_flags:
                    new_db_flags_ref.append(db_flags_ref[mask][chunk_idx[0]])
                
                # Merge Metadata for logging
                first_idx = mask.nonzero(as_tuple=True)[0][chunk_idx[0]].item()
                all_paths_m, all_paths_r = [], []
                for c_idx in chunk_idx:
                    abs_idx = mask.nonzero(as_tuple=True)[0][c_idx].item()
                    all_paths_m.extend(data['frame_paths'][abs_idx])
                    all_paths_r.extend(data['ref_frame_paths'][abs_idx])

                merged_meta_list.append({
                    'main_name': data['name'][first_idx],
                    'ref_name': data['ref_name'][first_idx],
                    'dataset_name': data['dataset_name'][first_idx],
                    'frame_paths': all_paths_m,
                    'ref_frame_paths': all_paths_r,
                    'aug_params': data['aug_params'][first_idx],
                    'ref_aug_params': data['ref_aug_params'][first_idx],
                    'multiplier': data['multiplier'][first_idx] if 'multiplier' in data else 1,
                    'chunk_id': "merged"
                })

            # Final grouped tensors
            embs_main = torch.stack(new_embs_main)
            embs_ref = torch.stack(new_embs_ref)
            steps_main = torch.stack(new_steps_main)
            steps_ref = torch.stack(new_steps_ref)
            seq_lens_main = torch.stack(new_seq_lens_main)
            seq_lens_ref = torch.stack(new_seq_lens_ref)
            
            tokens_m, tokens_r = None, None
            if new_tokens_m:
                tokens_m = torch.stack(new_tokens_m)
                tokens_r = torch.stack(new_tokens_r)
            
            # Pre-compute gates while parameters are gathered in Forward pass
            gates = None
            if hasattr(self, 'gate') and self.gate is not None and tokens_m is not None:
                # tokens_m/r: (B_merged, T_merged, P, D)
                # Corresponding gates: Main as Query for MR, Ref as Query for RM
                gates_mr = self.gate(tokens_m, tokens_r)
                gates_rm = self.gate(tokens_r, tokens_m)
                gates = {
                    'mr': gates_mr,
                    'rm': gates_rm
                }

            return {
                'embs': torch.cat([embs_main, embs_ref], dim=0),
                'steps': torch.cat([steps_main, steps_ref], dim=0),
                'seq_lens': torch.cat([seq_lens_main, seq_lens_ref], dim=0),
                'gates': gates,
                'is_cut_masks': torch.stack(new_cut_masks_ref) if has_cut_masks else None,
                'has_dustbin': torch.stack(new_db_flags_ref) if has_db_flags else None,
                'merged_metadata': merged_meta_list,
                'raw_tokens': {
                    'mains_tokens': tokens_m,
                    'refs_tokens': tokens_r
                } if tokens_m is not None else None,
            }

        if raw_tokens_dict is not None:
            # Reshape tokens: (B, T, P, D)
            m_tokens = raw_tokens_dict['mains_tokens'].view(-1, num_steps, raw_tokens_dict['mains_tokens'].shape[-2], raw_tokens_dict['mains_tokens'].shape[-1])
            r_tokens = raw_tokens_dict['refs_tokens'].view(-1, num_steps, raw_tokens_dict['refs_tokens'].shape[-2], raw_tokens_dict['refs_tokens'].shape[-1])
            return {
                'embs': embs,
                'raw_tokens': {
                    'mains_tokens': m_tokens,
                    'refs_tokens': r_tokens
                },
            }

        return embs

    @abc.abstractmethod
    def compute_loss(self, embs, steps, seq_lens, global_step, training,
                     frame_labels=None, seq_labels=None, metadata=None):
        pass

    def train_one_iter(self, data, steps, seq_lens, global_step, optimizer):
        optimizer.zero_grad()
        
        device = next(self.parameters()).device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
        
        if 'ref_chosen_steps' in data and 'ref_seq_lens' in data:
            steps = torch.cat([data['chosen_steps'], data['ref_chosen_steps']], dim=0)
            seq_lens = torch.cat([data['seq_lens'], data['ref_seq_lens']], dim=0)
        
        steps = steps.to(device)
        seq_lens = seq_lens.to(device)
        
        embs = self.forward(data, steps, seq_lens, training=True)
        
        loss, loss_dict = self.compute_loss(embs, steps, seq_lens, global_step,
                                 training=True, frame_labels=data.get('frame_labels'),
                                 seq_labels=data.get('seq_labels'),
                                 metadata=data)
                                 
        loss.backward()
        optimizer.step()
        
        return loss, loss_dict

    def evaluate_one_iter(self, data, steps, seq_lens, global_step):
        device = next(self.parameters()).device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
        
        if 'ref_chosen_steps' in data and 'ref_seq_lens' in data:
            steps = torch.cat([data['chosen_steps'], data['ref_chosen_steps']], dim=0)
            seq_lens = torch.cat([data['seq_lens'], data['ref_seq_lens']], dim=0)
        
        steps = steps.to(device)
        seq_lens = seq_lens.to(device)
        
        with torch.no_grad():
            embs = self.forward(data, steps, seq_lens, training=False)
            loss, loss_dict = self.compute_loss(embs, steps, seq_lens, global_step,
                                     training=False, frame_labels=data.get('frame_labels'),
                                     seq_labels=data.get('seq_labels'),
                                     metadata=data)
        return loss, loss_dict
