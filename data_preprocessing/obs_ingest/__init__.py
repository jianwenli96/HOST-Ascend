"""Streaming OBS ingestion for the LeRobot converter.

Downloads ``task_XXX.tar.gz`` objects from a Huawei OBS bucket and feeds them
through the existing ``convert_lerobot`` pipeline (``convert_lerobot_dataset.py``)
as they arrive: download -> extract -> convert -> commit -> cleanup, with
resumable state so a 9 TB run survives restarts.  See
``obs_streaming_convert.py`` in the parent directory for the entry point.
"""
