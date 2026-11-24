"""Independent artifact acceptance must reject internally rehashed corruption."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from paper_d_rotation import perturb, verify
from paper_d_rotation.tests.test_perturb import disk_data


def rewrite_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def refresh_hash(folder, relative):
    path = folder / 'verification.json'
    receipt = json.loads(path.read_text())
    receipt['files'][relative] = verify._sha(folder / relative)
    path.write_text(json.dumps(receipt))


class IndependentAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        data, baseline = disk_data(cls.root)
        perturb.run(cls.root/'run', data, workers=1, baseline_root=baseline)
        cls.folder = cls.root/'run/d4'
        cls.original = {p.name: p.read_bytes() for p in cls.folder.iterdir() if p.is_file()}
        cls.counts = {0: 1, 1: 2, 2: 1, 4: 1}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tearDown(self):
        for name, content in self.original.items():
            (self.folder/name).write_bytes(content)

    def test_complete_saved_diagnostics_independently_recompute(self):
        result = verify.verify_d4(self.folder, self.counts)
        self.assertEqual(result['baseline_queries_per_protocol'], 5)
        self.assertEqual(result['eligible_queries_per_protocol'], 2)
        self.assertEqual(result['area_matched_queries_per_protocol'], 1)
        self.assertEqual(result['single_region_deletion_rows'], 12)
        self.assertEqual(result['summary_rows_recomputed'], 112)

    def test_random_metric_corruption_detected_even_when_hash_is_updated(self):
        path = self.folder/'strategies.csv'
        rows = verify._csv(path)
        row = next(r for r in rows if r['stage']=='test' and r['strategy']=='random')
        row['reference_rank'] = str(float(row['reference_rank']) + .25)
        rewrite_csv(path, rows); refresh_hash(self.folder, path.name)
        with self.assertRaisesRegex(ValueError, 'Strategy expectation'):
            verify.verify_d4(self.folder, self.counts)

    def test_high_low_selection_cannot_be_relabelled(self):
        path = self.folder/'strategies.csv'
        rows = verify._csv(path)
        row = next(r for r in rows if r['subset']=='area_matched' and r['strategy']=='high')
        row['selected_regions'] = '0|1'
        rewrite_csv(path, rows); refresh_hash(self.folder, path.name)
        with self.assertRaisesRegex(ValueError, 'region selection'):
            verify.verify_d4(self.folder, self.counts)

    def test_summary_with_wrong_query_denominator_is_rejected(self):
        path = self.folder/'summary.csv'
        rows = verify._csv(path)
        rows[0]['valid_queries'] = '999'
        rewrite_csv(path, rows); refresh_hash(self.folder, path.name)
        with self.assertRaisesRegex(ValueError, 'query counts'):
            verify.verify_d4(self.folder, self.counts)

    def test_masking_must_keep_original_denominator(self):
        path = self.folder/'deletions.csv'
        rows = verify._csv(path)
        rows[0]['deleted_weight'] = '1.0'
        rewrite_csv(path, rows); refresh_hash(self.folder, path.name)
        with self.assertRaisesRegex(ValueError, 'area weight'):
            verify.verify_d4(self.folder, self.counts)

    def test_completed_artifact_hash_change_is_rejected(self):
        path = self.folder/'summary.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify.verify_d4(self.folder, self.counts)


if __name__ == '__main__':
    unittest.main()
