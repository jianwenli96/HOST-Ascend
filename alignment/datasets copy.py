# coding=utf-8
"""Datasets."""

import os
import json
import glob
import tensorflow.compat.v2 as tf
import torch
import numpy as np
from config import CONFIG
from dataset_splits import DATASETS
from preprocessors import preprocess_sequence

# We need to enable eager execution for tf.data to work nicely if not already enabled
# tf.enable_eager_execution() # TF 2.0 is eager by default

def normalize_input(frame, new_max=1., new_min=0.0, old_max=255.0, old_min=0.0):
  x = tf.cast(frame, tf.float32)
  x = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
  return x


def preprocess_input(frames, augment=True):
  """Preprocesses raw frames and optionally performs data augmentation."""

  preprocessing_ranges = {
      preprocess_sequence.IMAGE_TO_FLOAT: (),
      preprocess_sequence.RESIZE: {
          'new_size': [CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE],
      },
      preprocess_sequence.CLIP: {
          'lower_limit': 0.0,
          'upper_limit': 1.0,
      },
      preprocess_sequence.NORMALIZE_MEAN_STDDEV: {
          'mean': 0.5,
          'stddev': 0.5,
      }
  }

  if augment:
    if CONFIG.AUGMENTATION.RANDOM_FLIP:
      preprocessing_ranges[preprocess_sequence.FLIP] = {
          'dim': 2,
          'probability': 0.5,
      }
    if CONFIG.AUGMENTATION.RANDOM_CROP:
      preprocessing_ranges[preprocess_sequence.RANDOM_CROP] = {
          'image_size': tf.shape(frames)[1:4],
          'min_scale': 0.8,
      }
    if CONFIG.AUGMENTATION.BRIGHTNESS:
      preprocessing_ranges[preprocess_sequence.BRIGHTNESS] = {
          'max_delta': CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA,
      }
    if CONFIG.AUGMENTATION.CONTRAST:
      preprocessing_ranges[preprocess_sequence.CONTRAST] = {
          'lower': CONFIG.AUGMENTATION.CONTRAST_LOWER,
          'upper': CONFIG.AUGMENTATION.CONTRAST_UPPER
      }
    if CONFIG.AUGMENTATION.HUE:
      preprocessing_ranges[preprocess_sequence.HUE] = {
          'max_delta': CONFIG.AUGMENTATION.HUE_MAX_DELTA,
      }
    if CONFIG.AUGMENTATION.SATURATION:
      preprocessing_ranges[preprocess_sequence.SATURATION] = {
          'lower': CONFIG.AUGMENTATION.SATURATION_LOWER,
          'upper': CONFIG.AUGMENTATION.SATURATION_UPPER
      }
  else:
    if CONFIG.AUGMENTATION.RANDOM_CROP:
      preprocessing_ranges[preprocess_sequence.CENTRAL_CROP] = {
          'image_size': tf.shape(frames)[1:3]
      }

  frames, = preprocess_sequence.preprocess_sequence(
      ((frames, preprocess_sequence.IMAGE),), preprocessing_ranges)
  return frames


def decode(serialized_example):
  """Decode serialized SequenceExample."""

  context_features = {
      'name': tf.io.FixedLenFeature([], dtype=tf.string),
      'len': tf.io.FixedLenFeature([], dtype=tf.int64),
      'label': tf.io.FixedLenFeature([], dtype=tf.int64),
  }

  seq_features = {}

  seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)

  if CONFIG.DATA.FRAME_LABELS:
    seq_features['frame_labels'] = tf.io.FixedLenSequenceFeature(
        [], dtype=tf.int64)

  # Extract features from serialized data.
  context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=seq_features)

  seq_len = context_data['len']
  seq_label = context_data['label']

  video = sequence_data.get('video', [])
  frame_labels = sequence_data.get('frame_labels', [])
  name = tf.cast(context_data['name'], tf.string)
  return video, frame_labels, seq_label, seq_len, name


def get_steps(step):
  """Sample multiple context steps for a given step."""
  num_steps = CONFIG.DATA.NUM_STEPS
  stride = CONFIG.DATA.FRAME_STRIDE
  if num_steps < 1:
    raise ValueError('num_steps should be >= 1.')
  if stride < 1:
    raise ValueError('stride should be >= 1.')
  # We don't want to encode information from the future.
  steps = tf.range(step - (num_steps - 1) * stride, step + stride, stride)
  return steps


def sample_and_preprocess(video,
                          labels,
                          seq_label,
                          seq_len,
                          name,
                          num_steps,
                          augment,
                          sample_all=False,
                          sample_all_stride=1,
                          add_shape=False):
  """Samples frames and prepares them for training."""

  if sample_all:
    steps = tf.range(0, seq_len, sample_all_stride)
    seq_len = tf.shape(steps)[0]
    chosen_steps = steps
  else:
    stride = CONFIG.DATA.STRIDE
    sampling_strategy = CONFIG.DATA.SAMPLING_STRATEGY

    if sampling_strategy == 'stride':
      offset = tf.random.uniform(
          (), 0, tf.maximum(tf.cast(1, tf.int64), seq_len - stride * num_steps),
          dtype=tf.int64)
      steps = tf.minimum(
          seq_len - 1,
          tf.range(offset, offset + num_steps * stride + 1, stride))
      steps = steps[:num_steps]
    elif sampling_strategy == 'offset_uniform':
      check1 = tf.debugging.assert_greater_equal(
          seq_len,
          tf.cast(CONFIG.DATA.RANDOM_OFFSET, tf.int64),
          message='Random offset is more than sequence length.')
      check2 = tf.less_equal(
          tf.cast(num_steps, tf.int64),
          seq_len - tf.cast(CONFIG.DATA.RANDOM_OFFSET, tf.int64),
      )

      def _sample_random():
        with tf.control_dependencies([tf.identity(check1.outputs[0])]):
          offset = CONFIG.DATA.RANDOM_OFFSET
          steps = tf.random.shuffle(tf.range(offset, seq_len))
          steps = tf.gather(steps, tf.range(0, num_steps))
          steps = tf.gather(steps,
                            tf.nn.top_k(steps, k=num_steps).indices[::-1])
          return steps

      def _sample_all():
        return tf.range(0, num_steps, dtype=tf.int64)

      steps = tf.cond(check2, _sample_random, _sample_all)

    else:
      raise ValueError('Sampling strategy %s is unknown. Supported values are '
                       'stride, offset_uniform .' % sampling_strategy)

    if not sample_all and 'tcn' in CONFIG.TRAINING_ALGO:
      pos_window = CONFIG.TCN.POSITIVE_WINDOW
      pos_steps = tf.map_fn(
          lambda step: tf.random.uniform((),
                                         minval=step - pos_window,
                                         maxval=step, dtype=tf.int64),
          steps)
      steps = tf.stack([pos_steps, steps])
      steps = tf.reshape(tf.transpose(steps), (-1,))

    # Store chosen indices.
    chosen_steps = steps
    # Get multiple context steps depending on config at selected steps.
    steps = tf.reshape(tf.map_fn(get_steps, steps), [-1])
    steps = tf.maximum(tf.cast(0, tf.int64), steps)
    steps = tf.minimum(seq_len - 1, steps)

  shape_all_steps = CONFIG.DATA.NUM_STEPS * num_steps
  if not sample_all and 'tcn' in CONFIG.TRAINING_ALGO:
    shape_all_steps *= 2

  # Select data based on steps/
  video = tf.gather(video, steps)
  # Decode the encoded JPEG images
  video = tf.map_fn(
      tf.image.decode_jpeg,
      video,
      parallel_iterations=60,
      dtype=tf.uint8)
  # Take images in range [0, 255] and normalize to [0, 1]
  video = tf.map_fn(
      normalize_input,
      video,
      parallel_iterations=60,
      dtype=tf.float32)
  # Perform data-augmentation and return images in range [-1, 1]
  video = preprocess_input(video, augment)
  if add_shape:
    video.set_shape([shape_all_steps, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, 3])

  if CONFIG.DATA.FRAME_LABELS:
    labels = tf.gather(labels, steps)
    if add_shape:
      labels.set_shape([shape_all_steps])

  return {
      'frames': video,
      'frame_labels': labels,
      'chosen_steps': chosen_steps,
      'seq_lens': seq_len,
      'seq_labels': seq_label,
      'name': name
  }


def get_tfrecords(dataset, split, path, per_class=False):
  """Get TFRecord files based on dataset and split."""

  if per_class:
    path_to_tfrecords = os.path.join(path % dataset, '*%s*'%split)
    tfrecord_files = sorted(tf.io.gfile.glob(path_to_tfrecords))
  else:
    path_to_tfrecords = os.path.join(path % dataset,
                                     '%s_%s*' % (dataset, split))

    tfrecord_files = sorted(tf.io.gfile.glob(path_to_tfrecords))

  if not tfrecord_files:
    raise ValueError('No tfrecords found at path %s' % path_to_tfrecords)

  return tfrecord_files

class TFDataIterator:
    def __init__(self, tf_dataset):
        self.tf_iterator = iter(tf_dataset)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        data = next(self.tf_iterator)
        # Convert to torch tensors
        torch_data = {}
        for k, v in data.items():
            if isinstance(v, tf.Tensor):
                v = v.numpy()
                if k in ['frames', 'ref_frames']:
                    # TF: (B, T, H, W, C) -> PyTorch: (B, T, C, H, W)
                    # But wait, v is (B, T, H, W, C)
                    # We want (B, T, C, H, W)
                    v = np.transpose(v, (0, 1, 4, 2, 3))
                
                if isinstance(v, np.ndarray):
                    if v.dtype == np.object_: # Strings
                         torch_data[k] = v # Keep as numpy array of objects or list
                    else:
                         torch_data[k] = torch.from_numpy(v)
                else:
                    torch_data[k] = v
            else:
                torch_data[k] = v
        return torch_data

def libero_generator():
    video_paths_file = '/x2robot_v2/ethanchen/code/tcc/video_paths.json'
    if not os.path.exists(video_paths_file):
        return

    with open(video_paths_file, 'r') as f:
        video_paths = json.load(f)
    
    # Infinite loop for training
    while True:
        np.random.shuffle(video_paths)
        for video_path in video_paths:
            task_paths_file = os.path.join(video_path, 'task_paths.json')
            if not os.path.exists(task_paths_file):
                continue
            
            try:
                with open(task_paths_file, 'r') as f:
                    task_paths = json.load(f)
            except:
                continue
            
            if 'same' not in task_paths or not task_paths['same']:
                continue
                
            # Randomly select a reference video
            ref_video_path = np.random.choice(task_paths['same'])
            
            # Yield for both views
            for view in ['images', 'gripper_images']:
                main_view_path = os.path.join(video_path, view)
                ref_view_path = os.path.join(ref_video_path, view)
                
                if not os.path.exists(main_view_path) or not os.path.exists(ref_view_path):
                    continue

                # Get sorted file lists
                def get_files(path):
                    files = glob.glob(os.path.join(path, '*.jpg'))
                    # Sort by numerical value of filename (0.jpg, 1.jpg, ...)
                    files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
                    return files

                main_files = get_files(main_view_path)
                ref_files = get_files(ref_view_path)
                
                if not main_files or not ref_files:
                    continue

                yield (main_files, ref_files, video_path, ref_video_path)

def _create_libero_dataset(preprocess_fn, batch_size):
    dataset = tf.data.Dataset.from_generator(
        libero_generator,
        output_types=(tf.string, tf.string, tf.string, tf.string),
        output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
    )
    
    def load_and_process(main_files, ref_files, main_name, ref_name):
        # Read files
        main_video = tf.map_fn(tf.io.read_file, main_files, dtype=tf.string)
        ref_video = tf.map_fn(tf.io.read_file, ref_files, dtype=tf.string)
        
        # Get lengths
        main_len = tf.shape(main_video)[0]
        ref_len = tf.shape(ref_video)[0]
        
        # Dummy labels
        main_labels = tf.zeros([main_len], dtype=tf.int64)
        ref_labels = tf.zeros([ref_len], dtype=tf.int64)
        seq_label = tf.cast(0, tf.int64)
        
        # Preprocess
        main_data = preprocess_fn(main_video, main_labels, seq_label, main_len, main_name)
        ref_data = preprocess_fn(ref_video, ref_labels, seq_label, ref_len, ref_name)
        
        # Merge
        combined = main_data
        for k, v in ref_data.items():
            combined['ref_' + k] = v
            
        return combined

    dataset = dataset.map(load_and_process, num_parallel_calls=60)
    
    # Batching
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    
    return dataset

def create_dataset(split, mode, batch_size=None, return_iterator=True):
  """Creates a single-class dataset iterator based on config and split."""

  per_class = CONFIG.DATA.PER_CLASS
  if mode == 'train':
    if not batch_size:
      batch_size = CONFIG.TRAIN.BATCH_SIZE
    num_steps = CONFIG.TRAIN.NUM_FRAMES
    preprocess_fn = (
        lambda video, labels, seq_label, seq_len, name: sample_and_preprocess(
            video,
            labels,
            seq_label,
            seq_len,
            name,
            num_steps,
            augment=True,
            add_shape=True))
  elif mode == 'eval':
    if not batch_size:
      batch_size = CONFIG.EVAL.BATCH_SIZE
    num_steps = CONFIG.EVAL.NUM_FRAMES
    preprocess_fn = (
        lambda video, labels, seq_label, seq_len, name: sample_and_preprocess(
            video,
            labels,
            seq_label,
            seq_len,
            name,
            num_steps,
            augment=False,
            add_shape=True))
  else:
    raise ValueError('Unidentified mode: %s. Use either train or eval.' % mode)

  fraction = CONFIG.DATA.PER_DATASET_FRACTION

  datasets = []
  with tf.device('/cpu:0'):
    for dataset_name in CONFIG.DATASETS:
      tfrecord_files = get_tfrecords(
          dataset_name, split, CONFIG.PATH_TO_TFRECORDS, per_class=per_class)
      dataset = tf.data.TFRecordDataset(
          tfrecord_files, num_parallel_reads=60)

      if (fraction != 1.0 and mode == 'train'):
        num_samples = max(1, int(fraction * DATASETS[dataset_name][split]))
        dataset = dataset.take(num_samples)
      else:
        num_samples = DATASETS[dataset_name][split]
      if CONFIG.DATA.SHUFFLE_QUEUE_SIZE <= 0:
        dataset = dataset.shuffle(num_samples)
      else:
        dataset = dataset.shuffle(CONFIG.DATA.SHUFFLE_QUEUE_SIZE)
      dataset = dataset.repeat()
      dataset = dataset.batch(batch_size)
      datasets.append(dataset)

    dataset = tf.data.experimental.sample_from_datasets(datasets,
                                                        len(datasets) * [1.0])

    dataset = dataset.unbatch()

    dataset = dataset.map(decode,
                          num_parallel_calls=60)

    dataset = dataset.map(preprocess_fn,
                          num_parallel_calls=60)

    # drop_remainder adds batch size in shape else first dim remains as None.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch batches
    dataset = dataset.prefetch(1)

    if return_iterator:
      return TFDataIterator(dataset)
    else:
      return dataset
