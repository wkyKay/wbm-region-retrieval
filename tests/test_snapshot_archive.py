"""No repository imports or mutations: synthetic byte-preservation regression tests."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from results_catalog import snapshot_archive as archive


class SnapshotArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.payload = (b'{"runs":[{"float":0.12345678901234567,"old_path":"/old/path"}],'
                        b'"unicode":"\xe6\x99\xb6\xe5\x9c\x86","null":null}\n  ')
        (self.folder / archive.RAW).write_bytes(self.payload)

    def files(self):
        return {p.name: p.read_bytes() for p in self.folder.iterdir() if p.is_file()}

    def test_pack_unpack_preserves_all_raw_bytes(self):
        raw = self.folder / archive.RAW
        stamp = raw.stat().st_mtime_ns
        metadata = archive.pack(self.folder)
        self.assertEqual(raw.read_bytes(), self.payload)
        self.assertEqual(raw.stat().st_mtime_ns, stamp)
        self.assertEqual(metadata[archive.RAW], {"sha256": hashlib.sha256(self.payload).hexdigest(), "size": len(self.payload)})
        raw.unlink()
        self.assertEqual(archive.unpack(self.folder)["status"], "unpacked")
        self.assertEqual(raw.read_bytes(), self.payload)
        self.assertEqual(archive.verify(self.folder)["status"], "verified")

    def test_deterministic_gzip_and_metadata_across_paths_and_times(self):
        archive.pack(self.folder)
        first = self.files()
        header = first[archive.ARCHIVE][:10]
        self.assertEqual(header[:4], b"\x1f\x8b\x08\x00")
        self.assertEqual(header[4:8], b"\x00" * 4)
        self.assertEqual(header[9], 255)
        with tempfile.TemporaryDirectory() as second:
            second = Path(second)
            (second / archive.RAW).write_bytes(self.payload)
            os.utime(second / archive.RAW, (1, 1))
            archive.pack(second)
            for name in (archive.ARCHIVE, archive.MANIFEST):
                self.assertEqual((second / name).read_bytes(), first[name])

    def test_pack_and_unpack_idempotent_preserve_mtime(self):
        archive.pack(self.folder)
        before = {p.name: p.stat().st_mtime_ns for p in self.folder.iterdir()}
        self.assertEqual(archive.pack(self.folder)["status"], "already_packed")
        self.assertEqual(archive.unpack(self.folder)["status"], "already_unpacked")
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in self.folder.iterdir()}, before)

    def test_existing_different_raw_never_overwritten(self):
        archive.pack(self.folder)
        (self.folder / archive.RAW).write_bytes(b'{"newer":true}')
        before = self.files()
        with self.assertRaisesRegex(ValueError, "differs"):
            archive.unpack(self.folder)
        self.assertEqual(self.files(), before)

    def test_raw_corruption_rejected_by_verify(self):
        archive.pack(self.folder)
        (self.folder / archive.RAW).write_bytes(self.payload + b" ")
        with self.assertRaisesRegex(ValueError, "differs"):
            archive.verify(self.folder)

    def test_gzip_corruption_rejected_without_partial_raw(self):
        archive.pack(self.folder)
        raw, compressed = self.folder / archive.RAW, self.folder / archive.ARCHIVE
        raw.unlink()
        compressed.write_bytes(compressed.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Compressed snapshot"):
            archive.unpack(self.folder)
        self.assertFalse(raw.exists())
        self.assertFalse(any(p.name.startswith(".snapshot_") for p in self.folder.iterdir()))

    def test_rebound_broken_gzip_still_rejected(self):
        archive.pack(self.folder)
        raw, compressed = self.folder / archive.RAW, self.folder / archive.ARCHIVE
        raw.unlink()
        compressed.write_bytes(compressed.read_bytes()[:-2])
        value = json.loads((self.folder / archive.MANIFEST).read_text())
        value[archive.ARCHIVE] = archive._fingerprint(compressed)
        (self.folder / archive.MANIFEST).write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "gzip"):
            archive.unpack(self.folder)
        self.assertFalse(raw.exists())

    def test_plain_size_and_hash_both_checked(self):
        archive.pack(self.folder)
        path = self.folder / archive.MANIFEST
        original = path.read_bytes()
        (self.folder / archive.RAW).unlink()
        for field, value in (("size", 1), ("sha256", "0" * 64)):
            with self.subTest(field=field):
                metadata = json.loads(original)
                metadata[archive.RAW][field] = value
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "Decoded snapshot"):
                    archive.unpack(self.folder)
                self.assertFalse((self.folder / archive.RAW).exists())

    def test_symlink_raw_refused_even_when_identical(self):
        archive.pack(self.folder)
        raw = self.folder / archive.RAW
        target = self.folder / "external.json"
        raw.rename(target)
        raw.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            archive.unpack(self.folder)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            archive.pack(self.folder)

    def test_pack_refreshes_archive_without_touching_new_raw(self):
        archive.pack(self.folder)
        raw = self.folder / archive.RAW
        raw.write_bytes(self.payload + b"\n")
        stamp = raw.stat().st_mtime_ns
        self.assertEqual(archive.pack(self.folder)["status"], "packed")
        self.assertEqual(raw.stat().st_mtime_ns, stamp)
        self.assertEqual(gzip.decompress((self.folder / archive.ARCHIVE).read_bytes()), self.payload + b"\n")

    def test_pack_pair_write_failure_restores_previous_archive(self):
        archive.pack(self.folder)
        (self.folder / archive.RAW).write_bytes(self.payload + b"\n")
        before = self.files()
        replace, failed = os.replace, False
        def fail_once(source, target):
            nonlocal failed
            if Path(target).name == archive.MANIFEST and not failed:
                failed = True
                raise OSError("injected metadata commit failure")
            return replace(source, target)
        with patch.object(archive.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):
                archive.pack(self.folder)
        self.assertEqual(self.files(), before)

    def test_racing_raw_creation_not_overwritten(self):
        archive.pack(self.folder)
        raw = self.folder / archive.RAW
        raw.unlink()
        link = os.link
        def race(source, destination):
            raw.write_bytes(b'{"newer":true}')
            return link(source, destination)
        with patch.object(archive.os, "link", side_effect=race):
            with self.assertRaisesRegex(ValueError, "differs"):
                archive.unpack(self.folder)
        self.assertEqual(raw.read_bytes(), b'{"newer":true}')


if __name__ == "__main__":
    unittest.main()
