# TCC PyTorch

This is a PyTorch implementation of Temporal Cycle-Consistency Learning (TCC).
It is a translation of the TensorFlow version.

## Requirements
- PyTorch
- TensorFlow (for data loading and preprocessing)
- Abseil (absl-py)
- PyYAML
- EasyDict
- Matplotlib

## Training
To train the model, run:
```bash
python train.py --logdir /path/to/your/logs
```

## Data
The code expects TFRecords as input, similar to the TF version.
Ensure `CONFIG.PATH_TO_TFRECORDS` in `config.py` points to the correct location.
