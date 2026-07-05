import h5py
import os
import json
import re
from typing import Dict, List
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import threading
import time
import argparse


class TaskDictionaryExtractor:
    def __init__(self, output_hdf5_path: str, groups: List[str] = None,
                 default_group: str = None, group_mapping: Dict[str, str] = None,
                 clear_existing: bool = False, json_output_path: str = None,
                 log_output_path: str = None, json_save_interval: int = 100):
        self.output_path = output_hdf5_path
        self.groups = groups or ['default']
        self.default_group = default_group or self.groups[0]
        self.group_mapping = group_mapping or {}

        base_path = Path(output_hdf5_path)
        self.json_output_path = json_output_path if json_output_path else str(base_path.with_suffix('.json'))
        self.log_output_path = log_output_path if log_output_path else str(base_path.with_name(base_path.stem + '_log.txt'))

        self.json_save_interval = json_save_interval
        self.json_save_counter = 0

        self.task_dict = defaultdict(lambda: defaultdict(lambda: {'same': []}))

        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'missing_instruction_files': 0,
            'empty_descriptions': 0,
            'errors': 0,
            'skipped_classes': 0,
            'skipped_folders': 0,
            'json_saves': 0,
            'last_json_save': None,
            'start_time': time.time(),
        }
        self.lock = threading.Lock()

        if clear_existing:
            self._clear_files()

        self._initialize_json()
        self._initialize_log()

    def _clear_files(self):
        for path, label in [(self.output_path, "HDF5"), (self.json_output_path, "JSON"), (self.log_output_path, "log")]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed existing {label} file: {path}")
                except Exception as e:
                    print(f"Failed to remove {label} file: {e}")

    def _initialize_json(self):
        if os.path.exists(self.json_output_path):
            try:
                with open(self.json_output_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    for group_name, tasks in loaded_data.items():
                        if group_name in self.groups:
                            for task_name, task_data in tasks.items():
                                self.task_dict[group_name][task_name] = task_data
                print("Loaded existing JSON checkpoint.")
            except Exception as e:
                print(f"Failed to load JSON checkpoint: {e}. Starting fresh.")
                self._save_json()
        else:
            self._save_json()

    def _initialize_log(self):
        with open(self.log_output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Task Dictionary Extractor (directory scan mode)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _save_json(self):
        try:
            json_data = {g: self.task_dict[g] for g in self.groups if g in self.task_dict}
            with open(self.json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            self.stats['json_saves'] += 1
            self.stats['last_json_save'] = time.strftime('%H:%M:%S')
        except Exception as e:
            print(f"Failed to save JSON: {e}")

    def _save_all_to_hdf5(self):
        print("\nWriting HDF5 file from in-memory task dictionary...")
        with self.lock:
            with h5py.File(self.output_path, 'w') as f:
                for group_name, tasks in self.task_dict.items():
                    group = f.create_group(group_name)
                    for task_name, task_data in tasks.items():
                        task_group = group.create_group(task_name)

                        existing_paths = task_data['same']
                        if not existing_paths:
                            continue

                        path_data = [p.encode('utf-8') for p in existing_paths]
                        dt = h5py.special_dtype(vlen=bytes)
                        ds = task_group.create_dataset('same', data=path_data, dtype=dt)
                        ds.attrs['count'] = len(existing_paths)
        print(f"HDF5 file written: {self.output_path}")

    def _update_log(self, message: str = None, force_full_update: bool = False):
        try:
            with open(self.log_output_path, 'a', encoding='utf-8') as f:
                if message:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
                if force_full_update:
                    self._write_full_stats(f)
        except Exception:
            pass

    def _write_full_stats(self, f):
        elapsed = time.time() - self.stats['start_time']
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Stats update ({int(elapsed)}s):\n")
        f.write(f"  Episodes processed: {self.stats['total_processed']}\n")
        f.write(f"  Successfully indexed: {self.stats['successful_extractions']}\n")
        f.write(f"  Skipped folders: {self.stats['skipped_folders']}\n")
        f.write(f"  Skipped classes: {self.stats['skipped_classes']}\n")
        f.write(f"  Missing instruction.json: {self.stats['missing_instruction_files']}\n")
        f.write(f"  Empty descriptions: {self.stats['empty_descriptions']}\n")
        f.write(f"  Errors: {self.stats['errors']}\n")
        f.write("=" * 40 + "\n\n")

    def _sanitize_task_name(self, task_name: str) -> str:
        if not task_name:
            return ""
        sanitized = task_name.replace('/', '_').replace('\\', '_')
        sanitized = ''.join(c.lower() if c.isalpha() else c for c in sanitized)
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        return sanitized.strip()

    @staticmethod
    def _extract_action_class(task_folder_name: str) -> str:
        """Derive action class from session folder name, e.g. pack_badminton -> pack badminton."""
        prefix_pattern = re.compile(r'^\d{8}-(?:day|night)-')
        suffix_pattern = re.compile(r'-\d+$')

        cleaned_name = prefix_pattern.sub('', task_folder_name)
        cleaned_name = suffix_pattern.sub('', cleaned_name)
        return cleaned_name.replace('_', ' ').replace('-', ' ').strip().lower()

    @staticmethod
    def _should_skip_folder(folder_name: str) -> bool:
        lowered = folder_name.lower()
        return 'test' in lowered or 'factory' in lowered

    @staticmethod
    def _format_distribute(target_dist) -> str:
        if not target_dist:
            return ""

        valid_tasks = []
        if isinstance(target_dist, dict):
            for v in target_dist.values():
                v_str = str(v).strip()
                if v_str:
                    if v_str.endswith('.'):
                        v_str = v_str[:-1]
                    valid_tasks.append(v_str)
        elif isinstance(target_dist, str) and target_dist.strip():
            v_str = target_dist.strip()
            if v_str.endswith('.'):
                v_str = v_str[:-1]
            valid_tasks.append(v_str)

        return ", ".join(valid_tasks) if valid_tasks else ""

    @staticmethod
    def _format_task_name(detailed_inst: str, joined_dist: str) -> str:
        if joined_dist:
            return f"Instruction: {detailed_inst}. Distributes: {joined_dist}."
        return f"Instruction: {detailed_inst}."

    def _collect_task_paths(self, input_dir: str) -> List[Path]:
        base_path = Path(input_dir)
        task_paths = []

        for task_entry in sorted(base_path.iterdir()):
            if not task_entry.is_dir():
                continue
            if self._should_skip_folder(task_entry.name):
                self.stats['skipped_folders'] += 1
                self._update_log(f"Skipped folder: {task_entry}")
                continue
            task_paths.append(task_entry)

        return task_paths

    def _parse_instruction_file(self, task_path: Path, action_class: str,
                                skip_classes_lower: List[str]) -> List[dict]:
        if action_class in skip_classes_lower:
            self.stats['skipped_classes'] += 1
            self._update_log(f"Skipped class: {action_class}")
            return []

        inst_path = task_path / "instruction.json"
        if not inst_path.exists():
            self.stats['missing_instruction_files'] += 1
            self._update_log(f"Missing instruction.json: {inst_path}")
            return []

        episodes = []
        try:
            with open(inst_path, 'r', encoding='utf-8') as f_inst:
                inst_data = json.load(f_inst)

            for run_id, task_info in inst_data.items():
                full_episode_path = str(task_path / run_id)

                detailed_inst = task_info.get("detailed_instruction", "")
                if isinstance(detailed_inst, str):
                    detailed_inst = detailed_inst.strip()
                    if detailed_inst.endswith('.'):
                        detailed_inst = detailed_inst[:-1]

                if not detailed_inst:
                    self.stats['empty_descriptions'] += 1
                    continue

                distribute = task_info.get("distribute", "")
                subtask_gen = task_info.get("subtask_generation", "")
                target_dist = distribute if distribute else subtask_gen
                joined_dist = self._format_distribute(target_dist)

                episodes.append({
                    'category': action_class,
                    'path': full_episode_path,
                    'formatted_task_name': self._format_task_name(detailed_inst, joined_dist),
                })
        except Exception as e:
            self.stats['errors'] += 1
            self._update_log(f"Parse error {inst_path}: {e}")

        return episodes

    def process_from_dir(self, input_dir: str, dataset_name: str, skip_classes: List[str] = None):
        base_path = Path(input_dir)
        if not base_path.exists() or not base_path.is_dir():
            print(f"Error: input directory not found: {input_dir}")
            return

        skip_classes = skip_classes or []
        skip_classes_lower = [c.lower().strip() for c in skip_classes]

        task_paths = self._collect_task_paths(input_dir)
        if not task_paths:
            print(f"Error: no session subdirectories found under {input_dir}")
            return

        print(f"Scanning directory: {input_dir}")
        print(f"Found {len(task_paths)} session folders. Parsing instruction.json files...")

        episodes_to_process = []
        for task_path in task_paths:
            action_class = self._extract_action_class(task_path.name)
            if not action_class:
                self.stats['skipped_folders'] += 1
                self._update_log(f"Could not parse action class, skipped: {task_path}")
                continue

            episodes_to_process.extend(
                self._parse_instruction_file(task_path, action_class, skip_classes_lower)
            )

        if not episodes_to_process:
            print("No valid episodes found. All entries may have been filtered out.")
            return

        print(f"Found {len(episodes_to_process)} valid episodes. Building task dictionary...")

        group_name = self.groups[0]

        with tqdm(total=len(episodes_to_process), desc=f"Processing {dataset_name}", unit="ep") as pbar:
            for item in episodes_to_process:
                self.stats['total_processed'] += 1

                ep_path = item['path']
                formatted_task_name = item['formatted_task_name']
                task_lower = self._sanitize_task_name(formatted_task_name)

                if ep_path not in self.task_dict[group_name][task_lower]['same']:
                    self.task_dict[group_name][task_lower]['same'].append(ep_path)

                self.stats['successful_extractions'] += 1
                pbar.set_postfix(task=task_lower[:20])

                # Periodically save JSON as a checkpoint
                self.json_save_counter += 1
                if self.json_save_counter >= self.json_save_interval:
                    self._save_json()
                    self._update_log(force_full_update=True)
                    self.json_save_counter = 0

                pbar.update(1)

        self._save_json()
        self._save_all_to_hdf5()
        self._update_log(force_full_update=True)

    def print_statistics(self):
        print("\n=== Processing Statistics ===")
        print(f"Episodes processed:       {self.stats['total_processed']}")
        print(f"Successfully indexed:     {self.stats['successful_extractions']}")
        print(f"Skipped folders:          {self.stats['skipped_folders']}")
        print(f"Skipped classes:          {self.stats['skipped_classes']}")
        print(f"Missing instruction.json: {self.stats['missing_instruction_files']}")
        print(f"Empty descriptions:       {self.stats['empty_descriptions']}")
        print(f"Errors:                   {self.stats['errors']}")

    def read_from_hdf5(self):
        print("\n=== HDF5 Output Verification ===")
        try:
            with h5py.File(self.output_path, 'r') as f:
                for group_name in f.keys():
                    grp = f[group_name]
                    print(f"Group: {group_name} ({len(grp)} unique task keys)")
                    for i, task in enumerate(grp.keys()):
                        if i >= 3:
                            break
                        count = grp[task]['same'].attrs.get('count', 0)
                        print(f"  [{task}]: {count} paths")
        except Exception as e:
            print(f"Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Build a task dictionary HDF5 from a raw robot data directory.'
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory of the raw dataset, e.g. /x2robot_data/zhengwei/10042')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output HDF5 path (companion .json and _log.txt are auto-generated)')
    parser.add_argument('--dataset_name', type=str, default='Egodex',
                        help='HDF5 group name (default: Egodex)')
    parser.add_argument('--clear', action='store_true',
                        help='Delete existing output files before running')
    parser.add_argument('--skip_classes', nargs='*', default=[],
                        help='Action classes to skip, e.g. --skip_classes "pack badminton"')

    args = parser.parse_args()

    extractor = TaskDictionaryExtractor(
        output_hdf5_path=args.output_path,
        groups=[args.dataset_name],
        clear_existing=args.clear,
    )

    extractor.process_from_dir(args.input_dir, args.dataset_name, skip_classes=args.skip_classes)
    extractor.print_statistics()
    extractor.read_from_hdf5()


if __name__ == "__main__":
    main()
