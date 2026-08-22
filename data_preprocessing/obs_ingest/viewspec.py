"""Run-level view spec resolution: which video features to convert.

The AgiBot tars ship both full-resolution AV1 videos
(``observation.images.head``, 480x640) and pre-resized 224x224 h264 copies
(``observation.images.head_compress``).  Converting from the ``_compress``
copies is ~10x cheaper (no scale filter, much faster h264 decode, ~8x smaller
JPEG output), but the choice must be uniform for the whole run: the converter
writes one global camera mapping / view list per dataset, so a per-tar
fallback would corrupt ``cam_mapping`` and the manifest's ``output_views``.

Resolution happens once, before any conversion, by probing the
``meta/info.json`` of the smallest few tars (only the compressed bytes up to
that member are streamed — never the videos section):

* all probed tars have every ``_compress`` variant -> compress views,
  ``output_size=None`` (already 224);
* otherwise -> the plain view keys plus the user's ``--output-size``.

A tar lacking the resolved view keys later fails permanently at conversion
with the converter's own "not a video feature" error.  The resolved spec is
stored in the pipeline state so a resumed run keeps the same views; explicit
``--main-view``/``--gripper-views`` flags skip probing entirely.
"""

from __future__ import annotations

from convert_lerobot.source import DEFAULT_GRIPPER_VIEWS, DEFAULT_MAIN_VIEW


def compress_variant(view_key: str) -> str:
    return view_key if view_key.endswith("_compress") else f"{view_key}_compress"


def info_has_video(info: dict, view_key: str) -> bool:
    features = info.get("features", {})
    feature = features.get(view_key)
    return isinstance(feature, dict) and feature.get("dtype") == "video"


def views_explicitly_set(args) -> bool:
    """True when the user passed --main-view/--gripper-views explicitly."""
    return args.main_view != DEFAULT_MAIN_VIEW or args.gripper_views != list(DEFAULT_GRIPPER_VIEWS)


def resolve_view_spec(probed_infos: list[dict], args) -> dict:
    """Resolve the run-level view spec from probed info.json dicts.

    ``probed_infos`` may be empty only when the view flags were set explicitly
    (callers skip probing in that case).  Returns a dict with ``main_view``,
    ``gripper_views``, ``output_size`` (``[height, width]`` or ``None``) and
    ``source`` (``"explicit"``, ``"compress"`` or ``"full_res"``).
    """
    if views_explicitly_set(args):
        return {
            "main_view": args.main_view,
            "gripper_views": list(args.gripper_views),
            "output_size": args.output_size,
            "source": "explicit",
        }
    if not probed_infos:
        raise RuntimeError(
            "View spec cannot be resolved: probing meta/info.json failed. Pass "
            "--main-view/--gripper-views explicitly to skip probing."
        )

    main_compress = compress_variant(DEFAULT_MAIN_VIEW)
    gripper_compress = [compress_variant(view) for view in DEFAULT_GRIPPER_VIEWS]
    all_have_compress = all(
        info_has_video(info, view_key)
        for info in probed_infos
        for view_key in (main_compress, *gripper_compress)
    )
    if all_have_compress:
        print(
            f"view spec: {len(probed_infos)} probed tars all ship *_compress 224x224 h264 "
            f"views; converting from {main_compress} / {', '.join(gripper_compress)} "
            f"without resizing (much faster than the 480x640 AV1 originals)"
        )
        return {
            "main_view": main_compress,
            "gripper_views": gripper_compress,
            "output_size": None,
            "source": "compress",
        }
    print(
        f"view spec: not all probed tars ship *_compress views; converting from the "
        f"full-resolution originals ({DEFAULT_MAIN_VIEW})"
        + (f" resized to {args.output_size}" if args.output_size else "")
    )
    return {
        "main_view": DEFAULT_MAIN_VIEW,
        "gripper_views": list(DEFAULT_GRIPPER_VIEWS),
        "output_size": args.output_size,
        "source": "full_res",
    }


def apply_view_spec(args, spec: dict) -> None:
    """Mutate the converter-facing argparse namespace with the resolved spec."""
    args.main_view = spec["main_view"]
    args.gripper_views = list(spec["gripper_views"])
    args.output_size = spec["output_size"]
