"""OBS access for the streaming pipeline: listing, ranged part downloads, probing.

The Huawei OBS SDK (``esdk-obs-python``) is imported lazily so this module
stays importable without it (synthetic tests inject their own client
factories).  The patterns follow the proven ``extract_agibot_tasks.py`` in
the sibling SDK checkout:

* one ``ObsClient`` per connection, recreated on reconnect (never shared
  across part threads, so a failed stream can be closed safely);
* ranged GETs via ``GetObjectHeader(range='bytes=S-')`` — the SDK prepends
  ``bytes=`` itself;
* ``loadStreamInMemory=False`` returns a lazy ``ResponseWrapper`` whose
  ``read()`` raises when the connection dies mid-body (the SDK only retries
  connection *opening*; mid-stream failures are handled here), which doubles
  as a free per-part integrity check: catch, reconnect at the consumed
  offset, continue.

Download resume strategy: each part is written with ``os.pwrite`` at its byte
offset into one preallocated sparse file, and a per-part marker is written
only after that part is complete and ``fsync``ed.  A re-run skips parts whose
markers are intact; an object that changed size invalidates every marker.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tarfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

CHUNK_SIZE = 1024 * 1024  # 1 MiB stream reads
MIN_PART_SIZE = 8 * 1024 * 1024  # never split an object below this per part
META_TASK_NAMES = ("info.json",)  # only meta/info.json is needed for probing


class OBSFatalError(RuntimeError):
    """The object cannot be fetched as expected (4xx, 200 on ranged GET, 416).

    Permanent per tar: retrying will not heal it, the part markers are
    invalidated by the caller.
    """


class StopRequested(Exception):
    """Raised by workers when the pipeline's stop event is set (SIGINT)."""


def load_infos(path: Path) -> dict:
    """Parse a key=value credentials file (OBS_AK/OBS_SK/OBS_ENDPOINT/OBS_BUCKET)."""
    infos: dict = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            infos[key.strip()] = value.strip()
    missing = [key for key in ("OBS_AK", "OBS_SK", "OBS_ENDPOINT", "OBS_BUCKET") if key not in infos]
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")
    return infos


def import_obs(obs_sdk_path: Optional[str] = None):
    """Import the OBS SDK, falling back to ``obs_sdk_path`` on sys.path."""
    try:
        import obs  # noqa: F401
    except ImportError:
        if obs_sdk_path:
            sys.path.insert(0, str(obs_sdk_path))
            try:
                import obs  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "The obs SDK is not importable (pip install esdk-obs-python, "
                    f"or check --obs-sdk-path {obs_sdk_path!r}). Original error: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                "The obs SDK is not importable. Install it (pip install esdk-obs-python) "
                "or pass --obs-sdk-path pointing at the SDK's src directory."
            )
    return obs


def list_tar_keys(client, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """Paginated listing of every ``.tar.gz`` object key under the prefix.

    ``delimiter="/"`` folds the extracted member objects (``task_XXX/...`` —
    the bucket also holds each tar's contents as individual objects) into
    ``commonPrefixs`` entries, so the listing returns only the objects
    directly under the prefix: the tars themselves.  Without the delimiter
    this scan walks hundreds of thousands of member objects.
    """
    keys: list[tuple[str, int]] = []
    marker = None
    while True:
        resp = client.listObjects(
            bucket, prefix=prefix, marker=marker, max_keys=1000, delimiter="/"
        )
        if resp.status >= 300:
            raise RuntimeError(
                f"listObjects failed: status={resp.status} code={resp.errorCode} "
                f"msg={resp.errorMessage}"
            )
        contents = resp.body.contents or []
        for entry in contents:
            if entry.key.lower().endswith(".tar.gz"):
                keys.append((entry.key, entry.size))
        if resp.body.is_truncated:
            marker = resp.body.next_marker if resp.body.next_marker else contents[-1].key
        else:
            break
    return keys


def load_tar_keys(
    client, bucket: str, prefix: str, cache_path: Path, refresh: bool
) -> list[tuple[str, int]]:
    """Listing with a JSON cache file, so a resume does not re-list the bucket."""
    if not refresh and cache_path.is_file():
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                keys = [(key, size) for key, size in json.load(file)]
            print(f"reused listing cache {cache_path}: {len(keys)} tar.gz objects")
            return keys
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    keys = list_tar_keys(client, bucket, prefix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump([[key, size] for key, size in keys], file)
    return keys


def make_stream_factory(
    infos: dict, bucket: str, key: str, timeout: int
) -> Callable[[int], tuple]:
    """Build ``factory(offset) -> (client, response_stream)`` for one object.

    The returned stream is the lazy response body starting at ``offset``
    compressed bytes.  Raises ``OBSFatalError`` for permanent object problems
    and ``RuntimeError`` for transient transport errors.
    """
    obs = import_obs()

    def factory(offset: int):
        client = obs.ObsClient(
            access_key_id=infos["OBS_AK"],
            secret_access_key=infos["OBS_SK"],
            server=infos["OBS_ENDPOINT"],
            timeout=timeout,
        )
        try:
            if offset == 0:
                resp = client.getObject(bucket, key, loadStreamInMemory=False)
            else:
                headers = obs.GetObjectHeader(range=f"bytes={offset}-")
                resp = client.getObject(bucket, key, loadStreamInMemory=False, headers=headers)
            if resp.status >= 300:
                client.close()
                if resp.status in (400, 401, 403, 404):
                    raise OBSFatalError(
                        f"getObject failed: HTTP {resp.status} {resp.errorCode}"
                    )
                raise RuntimeError(
                    f"getObject failed: HTTP {resp.status} {resp.errorCode} {resp.errorMessage}"
                )
            if offset > 0 and resp.status == 200:
                # The server ignored the Range header; writing at the offset
                # would corrupt the file. Treat as a changed/odd object.
                client.close()
                raise OBSFatalError(f"server ignored Range header for {key}")
            if resp.status not in (200, 206):
                client.close()
                raise OBSFatalError(f"unexpected status {resp.status} for {key}")
            return client, resp.body.response
        except OBSFatalError:
            raise
        except Exception as exc:
            try:
                client.close()
            except Exception:
                pass
            raise RuntimeError(
                f"transport error while opening stream at {offset}: {exc}"
            ) from exc

    return factory


class ResumableGzipReader:
    """File-like reader decompressing a gzip object served over ranged streams.

    When a connection dies mid-stream, it reconnects with
    ``Range: bytes=<consumed>-`` and keeps feeding the same zlib decompressor,
    so decompression continues seamlessly.  ``input_consumed`` tracks total
    compressed bytes fetched.  (Ported from extract_agibot_tasks.py in the
    OBS SDK checkout.)
    """

    def __init__(self, open_fn, chunk_size: int = CHUNK_SIZE, max_resumes: int = 10):
        self.open_fn = open_fn
        self.chunk_size = chunk_size
        self.max_resumes = max_resumes
        self.resume_count = 0
        self.input_consumed = 0
        self.buf = b""
        self.finished = False
        self.dec = zlib.decompressobj(zlib.MAX_WBITS | 16)  # gzip format
        self.client = None
        self.wrapper = None
        self._connect(0)

    def _connect(self, offset: int) -> None:
        if self.client is not None:
            try:
                self.wrapper.close()
            except Exception:
                pass
            try:
                self.client.close()
            except Exception:
                pass
        self.client, self.wrapper = self.open_fn(offset)

    def read(self, n: Optional[int] = None) -> bytes:
        while True:
            if self.finished:
                break
            if n is not None and len(self.buf) >= n:
                break
            try:
                raw = self.wrapper.read(self.chunk_size)
            except Exception as exc:
                self.resume_count += 1
                if self.resume_count > self.max_resumes:
                    self.finished = True
                    raise RuntimeError(
                        f"stream died after {self.max_resumes} resumes: {exc}"
                    ) from exc
                print(
                    f"  [stream] connection died at byte {self.input_consumed} "
                    f"({exc}); resuming (attempt {self.resume_count})",
                    flush=True,
                )
                try:
                    self._connect(self.input_consumed)
                except Exception as exc2:
                    print(f"  [stream] reconnect failed: {exc2}", flush=True)
                continue
            if not raw:
                self.finished = True
                break
            self.input_consumed += len(raw)
            out = self.dec.decompress(raw)
            if out:
                self.buf += out
            if self.dec.eof:
                self.finished = True
                break
        if n is None:
            out, self.buf = self.buf, b""
        else:
            out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def close(self) -> None:
        self.finished = True
        if self.wrapper is not None:
            try:
                self.wrapper.close()
            except Exception:
                pass
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass


def probe_info_json(
    factory: Callable[[int], tuple], max_bytes: Optional[int] = None
) -> Optional[dict]:
    """Stream-decompress a tar.gz lazily and return its ``meta/info.json`` dict.

    Only the compressed bytes up to that member are fetched (meta follows data
    in these tars, and the videos section is never touched).  Returns ``None``
    if no info.json member is found within ``max_bytes`` compressed bytes.
    """
    reader = ResumableGzipReader(factory)
    tar = None
    try:
        tar = tarfile.open(fileobj=reader, mode="r|")
        for member in tar:
            parts = member.name.split("/")
            if (
                member.isfile()
                and len(parts) >= 2
                and parts[-2] == "meta"
                and parts[-1] == "info.json"
            ):
                return json.loads(tar.extractfile(member).read().decode("utf-8", "replace"))
            if max_bytes is not None and reader.input_consumed > max_bytes:
                return None
        return None
    finally:
        if tar is not None:
            tar.close()
        reader.close()


def plan_parts(size: int, parts: int) -> list[tuple[int, int]]:
    """Split [0, size) into up to ``parts`` non-empty byte ranges."""
    parts = max(1, min(parts, max(1, math.ceil(size / MIN_PART_SIZE))))
    boundaries = [size * index // parts for index in range(parts + 1)]
    return [
        (start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]


def marker_path(parts_dir: Path, tar_id: str, part_index: int) -> Path:
    return parts_dir / f"{tar_id}.part.{part_index}"


def _write_marker(path: Path, part_start: int, part_end: int, object_size: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump({"start": part_start, "end": part_end, "size": object_size}, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _read_marker(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _download_part(
    fd: int,
    part_index: int,
    part_start: int,
    part_end: int,
    parts_dir: Path,
    tar_id: str,
    key: str,
    object_size: int,
    factory: Callable[[int], tuple],
    retries: int,
    stop_event: Optional[threading.Event],
    progress_cb: Optional[Callable[[int], None]],
) -> int:
    marker = marker_path(parts_dir, tar_id, part_index)
    existing = _read_marker(marker)
    if existing is not None and existing.get("size") == object_size:
        return 0  # this part completed on a previous run
    consumed = 0
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        client = None
        stream = None
        try:
            client, stream = factory(part_start + consumed)
            while consumed < part_end - part_start:
                if stop_event is not None and stop_event.is_set():
                    raise StopRequested()
                chunk = stream.read(min(CHUNK_SIZE, part_end - part_start - consumed))
                if not chunk:
                    # A well-behaved ResponseWrapper raises instead of returning
                    # b''; treat empty reads as a mid-stream death too.
                    raise RuntimeError("stream ended before the part boundary")
                os.pwrite(fd, chunk, part_start + consumed)
                consumed += len(chunk)
                if progress_cb is not None:
                    progress_cb(len(chunk))
            break
        except StopRequested:
            raise
        except OBSFatalError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(
                    f"  [part {part_index}] {key}: attempt {attempt} failed at "
                    f"{consumed}/{part_end - part_start} bytes ({exc}); retrying in "
                    f"{min(2 ** attempt, 30)}s",
                    flush=True,
                )
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(
                f"part {part_index} of {key} failed after {retries} attempts "
                f"({last_error}); {consumed}/{part_end - part_start} bytes written"
            ) from last_error
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
    os.fsync(fd)
    _write_marker(marker, part_start, part_end, object_size)
    return consumed


def download_tar(
    tar_path: Path,
    parts_dir: Path,
    tar_id: str,
    key: str,
    size: int,
    factory: Callable[[int], tuple],
    parts: int,
    retries: int,
    stop_event: Optional[threading.Event] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> int:
    """Download one object into a preallocated sparse file via ranged parts.

    Parts whose markers are intact are skipped, so an interrupted download
    resumes at byte granularity.  Returns the number of bytes fetched by this
    invocation.  Raises ``OBSFatalError`` for permanent object problems.
    """
    plan = plan_parts(size, parts)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(tar_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, size)  # sparse: no zero-fill write
        if len(plan) == 1:
            fetched = _download_part(
                fd, 0, plan[0][0], plan[0][1], parts_dir, tar_id, key, size,
                factory, retries, stop_event, progress_cb,
            )
        else:
            with ThreadPoolExecutor(max_workers=len(plan)) as pool:
                futures = [
                    pool.submit(
                        _download_part,
                        fd, index, start, end, parts_dir, tar_id, key, size,
                        factory, retries, stop_event, progress_cb,
                    )
                    for index, (start, end) in enumerate(plan)
                ]
                fetched = 0
                for future in futures:
                    fetched += future.result()  # re-raise the first failure
        os.fsync(fd)
        return fetched
    finally:
        os.close(fd)
