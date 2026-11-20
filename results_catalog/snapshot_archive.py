#!/usr/bin/env python3
"""Lossless storage for the catalogue snapshot; existing readers keep using JSON.

Git stores snapshot.json.gz plus snapshot.sha256 (a JSON checksum manifest).
After a fresh clone: unpack, then run the existing build.py render/check.
No JSON parsing or serialization is performed on snapshot.json itself.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

RAW = "snapshot.json"
ARCHIVE = "snapshot.json.gz"
MANIFEST = "snapshot.sha256"
BLOCK = 1024 * 1024


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, non-symlink file: " + str(path))
    return path


def _fingerprint(path):
    digest, count = hashlib.sha256(), 0
    with _regular(path).open("rb") as handle:
        while block := handle.read(BLOCK):
            digest.update(block)
            count += len(block)
    return {"sha256": digest.hexdigest(), "size": count}


def _metadata(folder):
    value = json.loads(_regular(Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != {"schema_version", "compression", RAW, ARCHIVE}
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("compression") != {"format": "gzip", "level": 9, "mtime": 0, "filename": ""}):
        raise ValueError("Invalid snapshot checksum manifest")
    for name in (RAW, ARCHIVE):
        item = value[name]
        if (not isinstance(item, dict) or set(item) != {"sha256", "size"}
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or type(item.get("size")) is not int or item["size"] < 0):
            raise ValueError("Invalid snapshot hash or byte count: " + name)
    return value


@contextmanager
def _lock(folder):
    """Exclude simultaneous archive commands, without changing catalogue code."""
    path = Path(folder) / ".snapshot_archive.lock"
    try:
        path.mkdir()
    except FileExistsError as error:
        raise ValueError("Snapshot archive command already locked: " + str(path)) from error
    try:
        yield
    finally:
        path.rmdir()


def _decode(archive, metadata, *, destination=None, compare=None):
    """Validate compressed bytes, gzip integrity and every decoded byte."""
    if _fingerprint(archive) != metadata[ARCHIVE]:
        raise ValueError("Compressed snapshot hash or size mismatch")
    digest, size = hashlib.sha256(), 0
    try:
        with gzip.open(archive, "rb") as handle:
            while block := handle.read(BLOCK):
                size += len(block)
                if size > metadata[RAW]["size"]:
                    raise ValueError("Decoded snapshot exceeds recorded size")
                digest.update(block)
                if destination is not None:
                    destination.write(block)
                if compare is not None and compare.read(len(block)) != block:
                    raise ValueError("Existing snapshot.json differs; refusing to overwrite")
    except (OSError, EOFError) as error:
        raise ValueError("Invalid or truncated gzip snapshot") from error
    if {"sha256": digest.hexdigest(), "size": size} != metadata[RAW]:
        raise ValueError("Decoded snapshot hash or size mismatch")
    if compare is not None and compare.read(1):
        raise ValueError("Existing snapshot.json differs; refusing to overwrite")


def _verify(folder, metadata):
    raw = _regular(Path(folder) / RAW)
    with raw.open("rb") as handle:
        _decode(Path(folder) / ARCHIVE, metadata, compare=handle)
    # Also detect a concurrent raw-file replacement during the stream comparison.
    if _fingerprint(raw) != metadata[RAW]:
        raise ValueError("Snapshot changed during verification")


def verify(folder):
    """Require raw JSON and prove exact equality to the hash-checked archive."""
    folder = Path(folder)
    with _lock(folder):
        metadata = _metadata(folder)
        _verify(folder, metadata)
    return {"status": "verified", **metadata}


def pack(folder):
    """Archive the current raw bytes, leaving the raw file completely untouched.

    Archive and metadata replacements are atomic per file, with rollback for
    ordinary exceptions. A process/power loss between them fails later verify;
    the untouched raw JSON remains the authoritative recoverable source.
    """
    folder = Path(folder)
    with _lock(folder), tempfile.TemporaryDirectory(prefix=".snapshot_pack_", dir=folder) as temp:
        temp = Path(temp)
        raw = _regular(folder / RAW)
        for name in (ARCHIVE, MANIFEST):
            path = folder / name
            if path.exists() or path.is_symlink():
                _regular(path)
        original = _fingerprint(raw)
        compressed = temp / ARCHIVE
        with raw.open("rb") as source, compressed.open("wb") as target:
            # Empty filename prevents path-dependent FNAME; mtime=0 removes time.
            # GzipFile emits OS=255 rather than gzip.compress's platform header.
            with gzip.GzipFile(fileobj=target, mode="wb", filename="", compresslevel=9, mtime=0) as stream:
                shutil.copyfileobj(source, stream, BLOCK)
            target.flush()
            os.fsync(target.fileno())
        metadata = {"schema_version": 1,
                    "compression": {"format": "gzip", "level": 9, "mtime": 0, "filename": ""},
                    RAW: original, ARCHIVE: _fingerprint(compressed)}
        (temp / MANIFEST).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with raw.open("rb") as source:
            _decode(compressed, metadata, compare=source)
        if _fingerprint(raw) != original:
            raise ValueError("Snapshot changed during compression; no archive committed")
        if all((folder / name).is_file() and (folder / name).read_bytes() == (temp / name).read_bytes()
               for name in (ARCHIVE, MANIFEST)):
            return {"status": "already_packed", **metadata}
        backups, written = {}, []
        for name in (ARCHIVE, MANIFEST):
            path = folder / name
            backups[name] = path.exists()
            if backups[name]:
                shutil.copyfile(path, temp / (name + ".before"))
        try:
            for name in (ARCHIVE, MANIFEST):
                os.replace(temp / name, folder / name)
                written.append(name)
            _verify(folder, metadata)
        except BaseException:
            for name in reversed(written):
                if backups[name]:
                    os.replace(temp / (name + ".before"), folder / name)
                else:
                    (folder / name).unlink(missing_ok=True)
            raise
    return {"status": "packed", **metadata}


def unpack(folder):
    """Create missing raw JSON, or accept byte-identical raw JSON; never replace it."""
    folder = Path(folder)
    with _lock(folder):
        metadata = _metadata(folder)
        raw = folder / RAW
        if raw.exists() or raw.is_symlink():
            _verify(folder, metadata)
            return {"status": "already_unpacked", **metadata}
        descriptor, temporary = tempfile.mkstemp(prefix=".snapshot_unpack_", dir=folder)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                _decode(folder / ARCHIVE, metadata, destination=destination)
                destination.flush()
                os.fsync(destination.fileno())
            try:
                # Same-filesystem hard link is atomic and fails if raw appeared.
                # os.replace would silently overwrite a newer collected snapshot.
                os.link(temporary, raw)
            except FileExistsError:
                _verify(folder, metadata)
                return {"status": "already_unpacked", **metadata}
            _verify(folder, metadata)
        finally:
            temporary.unlink(missing_ok=True)
    return {"status": "unpacked", **metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "unpack", "verify"))
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    result = {"pack": pack, "unpack": unpack, "verify": verify}[args.command](args.catalog)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
