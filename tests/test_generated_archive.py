"""Byte-preservation and failure-safety checks using synthetic generated CSVs."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from results_catalog import generated_archive as archive


class GeneratedArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        (self.folder / "tables").mkdir()
        self.payloads = {
            "metrics.csv": b'key,value,text\r\na,0.12345678901234567,"line1\nline2"\r\n',
            "tables/fusion.csv": b'key,value\n\xe6\x99\xb6\xe5\x9c\x86,NaN\ntrailing,\n  ',
        }
        for name, payload in self.payloads.items():
            (self.folder / name).write_bytes(payload)

    def snapshot(self):
        return {str(p.relative_to(self.folder)):p.read_bytes() for p in self.folder.rglob("*") if p.is_file()}

    def metadata(self):
        return json.loads((self.folder / archive.MANIFEST).read_text())

    def write_metadata(self, value):
        (self.folder / archive.MANIFEST).write_text(json.dumps(value))

    def test_roundtrip_preserves_both_csvs_every_byte_and_raw_mtimes(self):
        before = {name:(self.folder / name).stat().st_mtime_ns for name in archive.RAW_FILES}
        packed = archive.pack(self.folder)
        for name, payload in self.payloads.items():
            self.assertEqual((self.folder / name).read_bytes(), payload)
            self.assertEqual((self.folder / name).stat().st_mtime_ns, before[name])
            self.assertEqual(packed["files"][name]["raw"],
                             {"sha256":hashlib.sha256(payload).hexdigest(), "size":len(payload)})
            (self.folder / name).unlink()
        self.assertEqual(archive.unpack(self.folder)["status"], "unpacked")
        for name, payload in self.payloads.items():
            self.assertEqual((self.folder / name).read_bytes(), payload)
        self.assertEqual(archive.verify(self.folder)["status"], "verified")

    def test_gzip_and_metadata_are_deterministic_across_locations_and_mtimes(self):
        archive.pack(self.folder)
        with tempfile.TemporaryDirectory() as other:
            other = Path(other); (other / "tables").mkdir()
            for name, payload in self.payloads.items():
                (other / name).write_bytes(payload); os.utime(other / name, (1, 1))
            archive.pack(other)
            for name in (*archive.ARCHIVES, archive.MANIFEST):
                self.assertEqual((self.folder / name).read_bytes(), (other / name).read_bytes())
            for name in archive.ARCHIVES:
                header = (self.folder / name).read_bytes()[:10]
                self.assertEqual(header[:8], b"\x1f\x8b\x08\x00" + b"\0" * 4)
                self.assertEqual(header[9], 255)

    def test_idempotent_commands_preserve_every_mtime(self):
        archive.pack(self.folder)
        before = {str(p):p.stat().st_mtime_ns for p in self.folder.rglob("*") if p.is_file()}
        self.assertEqual(archive.pack(self.folder)["status"], "already_packed")
        self.assertEqual(archive.unpack(self.folder)["status"], "already_unpacked")
        self.assertEqual(before, {str(p):p.stat().st_mtime_ns for p in self.folder.rglob("*") if p.is_file()})

    def test_streaming_operations_do_not_read_entire_csv_or_gzip(self):
        (self.folder / "metrics.csv").write_bytes(self.payloads["metrics.csv"] * 60000)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("nonstreaming read")):
            archive.pack(self.folder)
            for name in archive.RAW_FILES:
                (self.folder / name).unlink()
            archive.unpack(self.folder)
            archive.verify(self.folder)

    def test_pack_requires_both_generated_csvs_before_writing_archives(self):
        (self.folder / archive.RAW_FILES[1]).unlink()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "regular"):
            archive.pack(self.folder)
        self.assertEqual(self.snapshot(), before)

    def test_unpack_prechecks_both_tables_before_creating_missing_first_csv(self):
        archive.pack(self.folder)
        (self.folder / archive.RAW_FILES[0]).unlink()
        (self.folder / archive.RAW_FILES[1]).write_bytes(b"newer different CSV")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "differs"):
            archive.unpack(self.folder)
        self.assertEqual(self.snapshot(), before)

    def test_corrupt_second_gzip_does_not_restore_partial_first_csv(self):
        archive.pack(self.folder)
        for name in archive.RAW_FILES:
            (self.folder / name).unlink()
        path = self.folder / archive.ARCHIVES[1]
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Compressed table"):
            archive.unpack(self.folder)
        self.assertTrue(all(not (self.folder / name).exists() for name in archive.RAW_FILES))
        self.assertFalse(any(p.name.startswith(".generated_") for p in self.folder.iterdir()))

    def test_rebound_truncated_gzip_still_rejected(self):
        archive.pack(self.folder)
        raw = archive.RAW_FILES[1]; path = self.folder / (raw + ".gz")
        (self.folder / raw).unlink()
        path.write_bytes(path.read_bytes()[:-3])
        metadata = self.metadata(); metadata["files"][raw]["gzip"] = archive._fingerprint(path)
        self.write_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "gzip"):
            archive.unpack(self.folder)
        self.assertFalse((self.folder / raw).exists())

    def test_decoded_hash_and_size_are_both_checked(self):
        archive.pack(self.folder); original = self.metadata()
        raw = archive.RAW_FILES[0]; (self.folder / raw).unlink()
        for key, value in (("sha256", "0" * 64), ("size", 1)):
            with self.subTest(key=key):
                metadata = json.loads(json.dumps(original)); metadata["files"][raw]["raw"][key] = value
                self.write_metadata(metadata)
                with self.assertRaisesRegex(ValueError, "Decoded table"):
                    archive.unpack(self.folder)
                self.assertFalse((self.folder / raw).exists())

    def test_manifest_requires_exact_two_fixed_names_and_rejects_escape(self):
        archive.pack(self.folder); original = self.metadata()
        for change in ("missing", "escape", "extra", "bad_size"):
            with self.subTest(change=change):
                metadata = json.loads(json.dumps(original))
                if change == "missing":del metadata["files"][archive.RAW_FILES[1]]
                elif change == "escape":metadata["files"]["../outside.csv"] = metadata["files"].pop(archive.RAW_FILES[1])
                elif change == "extra":metadata["files"]["other.csv"] = metadata["files"][archive.RAW_FILES[1]]
                else:metadata["files"][archive.RAW_FILES[0]]["raw"]["size"] = True
                self.write_metadata(metadata)
                with self.assertRaises(ValueError):archive.verify(self.folder)

    def test_raw_corruption_is_rejected_and_never_overwritten(self):
        archive.pack(self.folder)
        path = self.folder / archive.RAW_FILES[0]; path.write_bytes(b"different raw")
        before = self.snapshot()
        for command in (archive.verify, archive.unpack):
            with self.assertRaisesRegex(ValueError, "differs"):command(self.folder)
        self.assertEqual(self.snapshot(), before)

    def test_all_source_archive_and_manifest_symlinks_are_rejected(self):
        archive.pack(self.folder)
        for name in (*archive.RAW_FILES, *archive.ARCHIVES, archive.MANIFEST):
            with self.subTest(name=name):
                path = self.folder / name; target = self.folder / "external"
                path.rename(target); path.symlink_to(target)
                try:
                    for command in (archive.pack, archive.unpack, archive.verify):
                        with self.assertRaisesRegex(ValueError, "non-symlink"):command(self.folder)
                finally:path.unlink(); target.rename(path)

    def test_catalogue_and_tables_directory_symlinks_are_rejected(self):
        archive.pack(self.folder)
        with tempfile.TemporaryDirectory() as other:
            alias = Path(other) / "alias"; alias.symlink_to(self.folder, target_is_directory=True)
            for command in (archive.pack, archive.unpack, archive.verify):
                with self.assertRaisesRegex(ValueError, "non-symlink"):command(alias)
            tables = self.folder / "tables"; target = self.folder / "external_tables"
            tables.rename(target); tables.symlink_to(target, target_is_directory=True)
            for command in (archive.pack, archive.unpack, archive.verify):
                with self.assertRaisesRegex(ValueError, "non-symlink"):command(self.folder)

    def test_missing_archive_cannot_be_silently_omitted(self):
        archive.pack(self.folder); (self.folder / archive.ARCHIVES[1]).unlink()
        for command in (archive.verify, archive.unpack):
            with self.assertRaisesRegex(ValueError, "regular"):command(self.folder)

    def test_pack_refreshes_from_changed_generated_bytes_only(self):
        archive.pack(self.folder)
        name = archive.RAW_FILES[1]; payload = self.payloads[name] + b"\nnew,0.42\n"
        (self.folder / name).write_bytes(payload); before = (self.folder / name).stat().st_mtime_ns
        self.assertEqual(archive.pack(self.folder)["status"], "packed")
        self.assertEqual((self.folder / name).stat().st_mtime_ns, before)
        self.assertEqual(gzip.decompress((self.folder / (name + ".gz")).read_bytes()), payload)
        archive.verify(self.folder)

    def test_failed_pack_commit_rolls_back_both_archives_and_manifest(self):
        archive.pack(self.folder)
        (self.folder / archive.RAW_FILES[0]).write_bytes(b"new generated data")
        before = self.snapshot(); replace = os.replace; failed = False
        def fail_once(source, target):
            nonlocal failed
            if Path(target).name == archive.MANIFEST and not failed:
                failed = True; raise OSError("injected commit failure")
            return replace(source, target)
        with patch.object(archive.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected"):archive.pack(self.folder)
        self.assertEqual(self.snapshot(), before)

    def test_racing_second_csv_is_not_overwritten_and_first_creation_rolls_back(self):
        archive.pack(self.folder)
        for name in archive.RAW_FILES:(self.folder / name).unlink()
        link = os.link; second = self.folder / archive.RAW_FILES[1]
        def race(source, destination):
            if Path(destination) == second:second.write_bytes(b"newer writer")
            return link(source, destination)
        with patch.object(archive.os, "link", side_effect=race):
            with self.assertRaisesRegex(ValueError, "differs"):archive.unpack(self.folder)
        self.assertFalse((self.folder / archive.RAW_FILES[0]).exists())
        self.assertEqual(second.read_bytes(), b"newer writer")

    def test_rollback_preserves_later_inplace_edit_of_first_restored_csv(self):
        archive.pack(self.folder)
        first, second = [self.folder / name for name in archive.RAW_FILES]
        first.unlink(); second.unlink(); link = os.link
        def race(source, destination):
            if Path(destination) == second:
                first.write_bytes(b"later inplace edit")
                second.write_bytes(b"newer writer")
            return link(source, destination)
        with patch.object(archive.os, "link", side_effect=race):
            with self.assertRaisesRegex(ValueError, "differs"):archive.unpack(self.folder)
        self.assertEqual(first.read_bytes(), b"later inplace edit")
        self.assertEqual(second.read_bytes(), b"newer writer")


if __name__ == "__main__":
    unittest.main()
