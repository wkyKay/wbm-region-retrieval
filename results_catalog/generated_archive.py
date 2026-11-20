#!/usr/bin/env python3
"""Lossless Git storage for the two large generated catalogue CSV tables.

The existing readers retain metrics.csv and tables/fusion.csv. Git stores their
deterministic .gz files and generated.sha256; unpack restores the original bytes.
No CSV parsing, rewriting or changes to catalogue generation are performed.
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
import zlib

RAW_FILES = ("metrics.csv", "tables/fusion.csv")
ARCHIVES = tuple(name + ".gz" for name in RAW_FILES)
MANIFEST = "generated.sha256"
COMPRESSION = {"format": "gzip", "level": 9, "mtime": 0, "filename": ""}
BLOCK = 1024 * 1024


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, non-symlink file: " + str(path))
    return path


def _folder(folder):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Expected a non-symlink catalogue directory: " + str(folder))
    return folder.resolve()


def _path(folder, name, *, missing=False):
    if name not in (*RAW_FILES, *ARCHIVES, MANIFEST):
        raise ValueError("Unexpected generated archive path: " + str(name))
    path = folder / name
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("Expected a non-symlink table directory: " + str(path.parent))
    if not missing or path.exists() or path.is_symlink():
        _regular(path)
    return path


def _fingerprint(path):
    checksum, size = hashlib.sha256(), 0
    with _regular(path).open("rb") as handle:
        while block := handle.read(BLOCK):
            checksum.update(block)
            size += len(block)
    return {"sha256": checksum.hexdigest(), "size": size}


def _metadata(folder):
    value = json.loads(_path(folder, MANIFEST).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != {"schema_version", "compression", "files"}
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("compression") != COMPRESSION
            or not isinstance(value.get("files"), dict) or set(value["files"]) != set(RAW_FILES)):
        raise ValueError("Invalid generated checksum manifest; both fixed tables are required")
    for name, record in value["files"].items():
        if not isinstance(record, dict) or set(record) != {"raw", "gzip"}:
            raise ValueError("Invalid generated table record: " + name)
        for item in record.values():
            if (not isinstance(item, dict) or set(item) != {"sha256", "size"}
                    or not isinstance(item.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    or type(item.get("size")) is not int or item["size"] < 0):
                raise ValueError("Invalid generated table hash or byte count: " + name)
    return value


@contextmanager
def _lock(folder):
    path = folder / ".generated_archive.lock"
    try:
        path.mkdir()
    except FileExistsError as error:
        raise ValueError("Generated archive command already locked: " + str(path)) from error
    try:
        yield
    finally:
        path.rmdir()


def _decode(archive, record, *, destination=None, compare=None):
    if _fingerprint(archive) != record["gzip"]:
        raise ValueError("Compressed table hash or size mismatch: " + str(archive))
    checksum, size = hashlib.sha256(), 0
    try:
        with gzip.open(archive, "rb") as handle:
            while block := handle.read(BLOCK):
                size += len(block)
                if size > record["raw"]["size"]:
                    raise ValueError("Decoded table exceeds recorded size")
                checksum.update(block)
                if destination is not None:
                    destination.write(block)
                if compare is not None and compare.read(len(block)) != block:
                    raise ValueError("Existing CSV differs; refusing to overwrite")
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("Invalid or truncated gzip table: " + str(archive)) from error
    if {"sha256": checksum.hexdigest(), "size": size} != record["raw"]:
        raise ValueError("Decoded table hash or size mismatch")
    if compare is not None and compare.read(1):
        raise ValueError("Existing CSV differs; refusing to overwrite")
    if _fingerprint(archive) != record["gzip"]:
        raise ValueError("Compressed table changed during decoding")


def _verify_one(folder, name, record):
    raw = _path(folder, name)
    with raw.open("rb") as handle:
        _decode(_path(folder, name + ".gz"), record, compare=handle)
    if _fingerprint(raw) != record["raw"]:
        raise ValueError("CSV changed during verification: " + name)


def _verify(folder, metadata):
    for name in RAW_FILES:
        _verify_one(folder, name, metadata["files"][name])


def verify(folder):
    """Require both raw CSVs and prove byte equality with their checked archives."""
    folder = _folder(folder)
    with _lock(folder):
        metadata = _metadata(folder)
        _verify(folder, metadata)
    return {"status": "verified", **metadata}


def pack(folder):
    """Archive both generated CSVs without modifying them; roll back failed commits.

    Replacements are atomic per file. An interrupted process can leave an invalid
    archive set, which verify rejects; the raw CSVs remain recoverable unchanged.
    """
    folder = _folder(folder)
    outputs = (*ARCHIVES, MANIFEST)
    with _lock(folder), tempfile.TemporaryDirectory(prefix=".generated_pack_", dir=folder) as temporary:
        temporary = Path(temporary)
        (temporary / "tables").mkdir()
        for name in RAW_FILES:
            _path(folder, name)
        for name in outputs:
            _path(folder, name, missing=True)
        records = {}
        for name in RAW_FILES:
            raw, compressed = _path(folder, name), temporary / (name + ".gz")
            original = _fingerprint(raw)
            with raw.open("rb") as source, compressed.open("wb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", filename="", compresslevel=9, mtime=0) as stream:
                    shutil.copyfileobj(source, stream, BLOCK)
                target.flush()
                os.fsync(target.fileno())
            records[name] = {"raw": original, "gzip": _fingerprint(compressed)}
            with raw.open("rb") as source:
                _decode(compressed, records[name], compare=source)
        metadata = {"schema_version": 1, "compression": COMPRESSION, "files": records}
        with (temporary / MANIFEST).open("w", encoding="utf-8") as target:
            target.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        for name in RAW_FILES:
            if _fingerprint(_path(folder, name)) != records[name]["raw"]:
                raise ValueError("CSV changed during compression; no archives committed")
        if all(_path(folder, name, missing=True).exists()
               and _fingerprint(folder / name) == _fingerprint(temporary / name) for name in outputs):
            return {"status": "already_packed", **metadata}
        backups, written = {}, []
        for index, name in enumerate(outputs):
            path = _path(folder, name, missing=True)
            backups[name] = temporary / ("before_" + str(index)) if path.exists() else None
            if backups[name] is not None:
                shutil.copyfile(path, backups[name])
        try:
            for name in outputs:
                _path(folder, name, missing=True)
                os.replace(temporary / name, folder / name)
                written.append(name)
            _verify(folder, metadata)
        except BaseException:
            for name in reversed(written):
                if backups[name] is not None:
                    os.replace(backups[name], folder / name)
                else:
                    (folder / name).unlink(missing_ok=True)
            raise
    return {"status": "packed", **metadata}


def unpack(folder):
    """Restore only missing CSVs after validating both tables; never replace a CSV."""
    folder = _folder(folder)
    with _lock(folder), tempfile.TemporaryDirectory(prefix=".generated_unpack_", dir=folder) as temporary:
        temporary = Path(temporary)
        metadata = _metadata(folder)
        pending, installed = {}, []
        for index, name in enumerate(RAW_FILES):
            raw = _path(folder, name, missing=True)
            if raw.exists():
                _verify_one(folder, name, metadata["files"][name])
            else:
                target = temporary / str(index)
                with target.open("wb") as destination:
                    _decode(_path(folder, name + ".gz"), metadata["files"][name], destination=destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                pending[name] = target
        try:
            for name, temporary_raw in pending.items():
                raw = _path(folder, name, missing=True)
                try:
                    # Hard-link creation is atomic and cannot overwrite a racing writer.
                    os.link(temporary_raw, raw)
                    installed.append((raw, temporary_raw.stat(), metadata["files"][name]["raw"]))
                except FileExistsError:
                    _verify_one(folder, name, metadata["files"][name])
            _verify(folder, metadata)
        except BaseException:
            for raw, original, expected in reversed(installed):
                current = raw.lstat() if raw.exists() or raw.is_symlink() else None
                if current is not None and (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino):
                    # Preserve a later writer even when it edited our inode in place.
                    try:
                        unchanged = _fingerprint(raw) == expected
                    except (OSError, ValueError):
                        unchanged = False
                    if unchanged:
                        raw.unlink()
            raise
    return {"status": "unpacked" if installed else "already_unpacked", **metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "unpack", "verify"))
    parser.add_argument("--catalog", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    print(json.dumps({"pack": pack, "unpack": unpack, "verify": verify}[args.command](args.catalog),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
