"""Stage modules of the LeRobot v2 -> HOST episode converter.

The CLI entry point is ``data_preprocessing/convert_lerobot_dataset.py``; it
imports the four stages from this package:

* ``source``     — stage A: source discovery and task-unit planning
* ``media``      — stage B: ffmpeg/ffprobe frame decoding and JPEG writing
* ``trajectory`` — stage C: trajectory conversion and normalization
* ``outputs``    — stages D/E: sidecar writing and verification

The pipeline processes one task at a time (streaming-compatible ingestion)
and encodes the episodes of a task in parallel (``--workers``).
"""
