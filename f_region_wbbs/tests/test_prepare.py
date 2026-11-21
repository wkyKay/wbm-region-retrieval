"""Independent original-encoding checks and immutable regional point caches."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from b1_structure_fusion.descriptors import load_settings, soft_histogram
from experiments.proposed.f_segmentation.proposals import segment
from experiments.proposed.f_segmentation.tensor_voting import tensor_vote
from f_region_structure.descriptors import describe_regions
from f_region_wbbs import prepare as preparation
from wbm_retrieval.common import file_hash, read_json


def fixture():
    raw = np.ones((32, 32), dtype=np.uint8)
    raw[3:6, 3:6] = 2
    raw[3:6, 17:20] = 2
    raw[17:20, 3:6] = 2
    raw[17:20, 17:20] = 2
    raw[27, 10:16] = 2
    raw[28, 28] = 2
    symmetric = np.ones(raw.shape, dtype=np.uint8)
    symmetric[15, 13:18] = 2
    singleton = np.ones(raw.shape, dtype=np.uint8)
    singleton[7, 7] = 2
    empty = np.ones(raw.shape, dtype=np.uint8)
    maps = np.stack([raw, symmetric, singleton, empty])
    labels = np.stack([segment(x, "cc8", 5, 4)[0] for x in maps])
    rows = [describe_regions(x, y) for x, y in zip(maps, labels)]
    cache = {key: np.concatenate([row[key] for row in rows]) for key in preparation.DESCRIPTOR_FIELDS}
    counts = np.asarray([int(y.max()) for y in labels])
    offsets = np.r_[0, counts.cumsum()].astype(np.int64)
    h = np.random.default_rng(2026).normal(size=(int(offsets[-1]), 512))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    cache.update(map_ids=np.arange(4), offsets=offsets, features=h, area=cache["original_count"],
                 histograms=cache["radial_hist16"], owners=np.repeat(np.arange(4), counts))
    protocol = SimpleNamespace(ids={"valid": np.arange(4), "test": np.arange(4)})
    return maps, labels, cache, protocol


def build(tmp, *, audit_ids=(0, 1), workers=0, fixture_values=None):
    maps, labels, cache, protocol = fixture_values or fixture()
    source = Path(tmp)/"frozen_source.json"
    source.write_text('{"frozen":true}')
    binding = {str(source): file_hash(source)}
    result = preparation.prepare(Path(tmp)/"new", maps, protocol, cache, labels, "context", binding,
                                 workers=workers, audit_ids=np.asarray(audit_ids, dtype=np.int64))
    return result, binding, (maps, labels, cache, protocol)


class PointFieldTests(unittest.TestCase):
    def test_one_vote_per_original_wafer_and_global_original_wbbs_values(self):
        maps, labels, _, _ = fixture()
        original = preparation.original_module()
        with patch("f_region_wbbs.prepare.tensor_vote", wraps=tensor_vote) as vote:
            fields = preparation._point_fields(maps[0], labels[0], load_settings())
        self.assertEqual(vote.call_count, 1)
        self.assertEqual(len(fields["descriptors"]), int((maps[0]==2).sum()))
        # threshold=0 retains ALL original points; all channel maxima remain global.
        all_original = original.encode_wafer(maps[0], {"noise_threshold": 0.})
        denoised_original = original.encode_wafer(maps[0])
        np.testing.assert_array_equal(fields["coordinates"], all_original.points)
        np.testing.assert_array_equal(fields["descriptors"], all_original.descriptors)
        np.testing.assert_array_equal(fields["coordinates"][fields["keep"]], denoised_original.points)
        np.testing.assert_array_equal(fields["descriptors"][fields["keep"]], denoised_original.descriptors)

    def test_joint_axis_reorder_reproduces_original_both_modes(self):
        maps, labels, cache, _ = fixture()
        row = preparation._job((maps[0], labels[0], load_settings(), True))
        for i, region in enumerate(row["rows"]):
            np.testing.assert_allclose(region["joint_all"], cache["joint_all"][i], rtol=0, atol=1e-14)
            np.testing.assert_allclose(region["joint"], cache["joint"][i], rtol=0, atol=1e-14)
            expected = soft_histogram(region["descriptors"][:, [2, 0, 1]]).reshape(-1)
            np.testing.assert_array_equal(region["joint_all"], expected)
        self.assertEqual(row["reference_audit"]["max_descriptor_error"], 0.)
        self.assertTrue(row["reference_audit"]["bitwise_descriptors_identical"])

    def test_masks_are_complete_original_top_four_with_source_raster_indices(self):
        maps, labels, _, _ = fixture()
        result = preparation._job((maps[0], labels[0], load_settings(), False))
        all_coordinates = np.argwhere(maps[0]==2)
        self.assertEqual(len(result["rows"]), 4)
        for token, region in enumerate(result["rows"], 1):
            expected = np.argwhere(labels[0]==token)
            np.testing.assert_array_equal(region["coordinates"], expected)
            np.testing.assert_array_equal(all_coordinates[region["source_point_indices"]], expected)
        self.assertIsNone(result["reference_audit"])
        self.assertGreater(len(all_coordinates), sum(len(row["coordinates"]) for row in result["rows"]))

    def test_singleton_zero_saliency_channels_and_zero_radius(self):
        raw = np.array([[2]], dtype=np.uint8)
        labels = np.array([[0]], dtype=np.uint8)
        fields = preparation._point_fields(raw, labels, load_settings())
        np.testing.assert_array_equal(fields["descriptors"], np.zeros((1, 3)))
        np.testing.assert_array_equal(fields["keep"], [True])
        result = preparation._job((raw, labels, load_settings(), True))
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["reference_audit"]["points"], 1)

    def test_empty_defect_map_does_not_vote_or_create_regions(self):
        raw = np.ones((12, 12), dtype=np.uint8)
        with patch("f_region_wbbs.prepare.tensor_vote", side_effect=AssertionError("empty")):
            result = preparation._job((raw, np.zeros_like(raw), load_settings(), True))
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["global_original_count"], 0)
        self.assertEqual(result["reference_audit"]["points"], 0)

    def test_original_mask_mutation_swapped_order_and_invalid_raw_rejected(self):
        maps, labels, _, _ = fixture()
        changed = labels[0].copy();changed[3, 3]=0
        swapped = labels[0].copy();swapped[labels[0]==1]=2;swapped[labels[0]==2]=1
        for bad in (changed, swapped, labels[0].astype(float), labels[0][:-1]):
            with self.assertRaises(ValueError):
                preparation._point_fields(maps[0], bad, load_settings())
        for raw in (np.zeros((4, 4)), np.full((4, 4), 4), np.ones((0, 4))):
            with self.assertRaises(ValueError):
                preparation._point_fields(raw, np.zeros_like(raw, dtype=np.uint8), load_settings())

    def test_spawn_matches_serial_without_pickling_full_dataset(self):
        maps, labels, _, _ = fixture()
        job = (maps[0], labels[0], load_settings(), True)
        expected = preparation._job(job)
        actual = list(preparation._map_jobs(iter([job]), workers=1))[0]
        self.assertEqual(actual["reference_audit"], expected["reference_audit"])
        for left, right in zip(actual["rows"], expected["rows"]):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])


class CacheTests(unittest.TestCase):
    def test_complete_cache_bytes_modes_diagnostics_trace_and_readonly_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, (maps, labels, source, _) = build(tmp)
            for key, value in source.items():
                self.assertEqual(result[key].dtype, value.dtype)
                self.assertEqual(result[key].tobytes(), value.tobytes(), key)
            self.assertEqual(result["region_labels"].tobytes(), labels.tobytes())
            self.assertEqual(tuple(result["point_sets"]), preparation.POINT_MODES)
            audit = read_json(result["folder"]/"audit.json")
            self.assertEqual(audit["production_tensor_vote_calls"], 3)
            self.assertEqual(audit["original_wbbs_audit_maps"], 2)
            self.assertEqual(audit["max_joint_reconstruction_error"], 0.)
            self.assertTrue(audit["old_masks_bitwise_identical"])
            for mode, pointset in result["point_sets"].items():
                self.assertEqual(pointset["descriptors"].dtype, np.float64)
                self.assertEqual(pointset["coordinates"].dtype, np.int64)
                for value in pointset.values():
                    self.assertIsInstance(value, np.memmap)
                    self.assertFalse(value.flags.writeable)
                counts = np.diff(pointset["offsets"])
                expected = source["area"] if mode=="all_points" else source["retained_count"]
                np.testing.assert_array_equal(counts, expected)
                diag = audit["point_sets"][mode]
                self.assertEqual(diag["empty_regions"], int((counts==0).sum()))
                self.assertEqual(diag["singleton_regions"], int((counts==1).sum()))
                duplicates = sum(int(y-x)-len(np.unique(pointset["descriptors"][x:y], axis=0))
                                 for x, y in zip(pointset["offsets"][:-1], pointset["offsets"][1:]))
                self.assertEqual(diag["duplicate_descriptor_rows"], duplicates)
            traces = [json.loads(line) for line in (result["folder"]/"region_trace.jsonl").read_text().splitlines()]
            for i, trace in enumerate(traces):
                self.assertEqual(trace["sample_id"], i)
                all_points = np.argwhere(maps[i]==2)
                for row in trace["regions"]:
                    expected = np.argwhere(labels[i]==row["token"])
                    np.testing.assert_array_equal(all_points[row["source_point_indices"]], expected)
            metadata = read_json(result["folder"]/"metadata.json")
            self.assertEqual(metadata["cnn_forward_passes"], 0)
            self.assertFalse(metadata["training_required"])
            self.assertLessEqual(metadata["started_at"], metadata["finished_at"])

    def test_complete_cache_reuses_without_vote_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, binding, (maps, labels, cache, protocol) = build(tmp)
            with patch("f_region_wbbs.prepare._job", side_effect=AssertionError("must reuse")):
                reused = preparation.prepare(Path(tmp)/"new", maps, protocol, cache, labels,
                    "context", binding, workers=0, audit_ids=[0, 1])
            self.assertEqual(result["features"].tobytes(), reused["features"].tobytes())
            changed = deepcopy(cache);changed["features"][0, 0] += .001
            for values, mask, audits in ((changed, labels, [0, 1]), (cache, labels[::-1], [0, 1]), (cache, labels, [0])):
                with self.assertRaises(ValueError):
                    preparation.prepare(Path(tmp)/"new", maps, protocol, values, mask,
                        "context", binding, workers=0, audit_ids=audits)

    def test_changed_source_binding_or_sealed_point_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, binding, (maps, labels, cache, protocol) = build(tmp)
            source = next(iter(binding));Path(source).write_text('changed')
            with self.assertRaises(ValueError):
                preparation.prepare(Path(tmp)/"new", maps, protocol, cache, labels,
                    "context", binding, workers=0, audit_ids=[0, 1])
            pointfile = result["folder"]/"all_points_descriptors.npy"
            with pointfile.open("r+b") as handle:
                handle.seek(-1, 2);byte=handle.read(1);handle.seek(-1, 2);handle.write(bytes([byte[0]^1]))
            with self.assertRaises(ValueError):
                preparation.read_cache(result["folder"])

    def test_resealed_wrong_original_coordinates_rejected_semantically(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _ = build(tmp)
            folder=result['folder']
            coordinates=np.load(folder/'all_points_coordinates.npy').copy()
            coordinates[0, 1] += 1
            np.save(folder/'all_points_coordinates.npy',coordinates)
            marker=read_json(folder/'completed.json')
            marker['files']['all_points_coordinates.npy']=file_hash(folder/'all_points_coordinates.npy')
            (folder/'completed.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'original region mask'):
                preparation.read_cache(folder)

    def test_changed_joint_reconstruction_rejected_before_completion(self):
        maps, labels, cache, protocol = fixture()
        cache["joint_all"][0, 0] += 1e-8
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'joint_all'):
                build(tmp, fixture_values=(maps, labels, cache, protocol))
            self.assertFalse((Path(tmp)/"new/cache/valid").exists())
            self.assertTrue(list((Path(tmp)/"new/cache").glob('.valid_attempt_*')))

    def test_incomplete_output_different_context_and_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, binding, (maps, labels, cache, protocol) = build(tmp)
            with self.assertRaises(ValueError):
                preparation.prepare(Path(tmp)/"new", maps, protocol, cache, labels,
                    "different-context", binding, workers=0, audit_ids=[0, 1])
            incomplete = Path(tmp)/"incomplete/cache/valid";incomplete.mkdir(parents=True)
            with self.assertRaises(ValueError):
                preparation.prepare(Path(tmp)/"incomplete", maps, protocol, cache, labels,
                    "context", binding, workers=0)
            cache["folder"] = Path(tmp)/"historical"
            with self.assertRaises(ValueError):
                preparation.prepare(Path(tmp)/"historical/new", maps, protocol, cache, labels,
                    "context", binding, workers=0)

    def test_no_region_cache_and_valid_test_separation(self):
        maps, labels, cache, protocol = fixture()
        maps=maps[2:];labels=labels[2:]
        for key in preparation.REQUIRED_FIELDS:
            if key not in ('map_ids','offsets'):cache[key]=cache[key][:0]
        cache.update(features=cache['features'][:0], map_ids=np.arange(2), offsets=np.zeros(3,dtype=np.int64))
        protocol=SimpleNamespace(ids={'valid':np.arange(2),'test':np.arange(2)})
        with tempfile.TemporaryDirectory() as tmp:
            result, binding, _=build(tmp, audit_ids=(0,1), fixture_values=(maps,labels,cache,protocol))
            self.assertEqual(result['features'].shape,(0,512))
            self.assertEqual(result['point_sets']['all_points']['descriptors'].shape,(0,3))
            test=preparation.prepare(Path(tmp)/'new',maps,protocol,cache,labels,'context',binding,
                                     workers=0,split='test',audit_ids=[0,1])
            self.assertNotEqual(test['folder'],result['folder'])
            self.assertNotEqual(read_json(test['folder']/'completed.json')['signature'],
                                read_json(result['folder']/'completed.json')['signature'])

    def test_invalid_shape_counts_workers_split_audit_and_source_rejected(self):
        maps, labels, cache, protocol = fixture()
        for field, value in [('map_ids', np.array([1,0,2,3])), ('offsets', np.array([0,5,5,5,5])),
                             ('features', cache['features'][:,:511]), ('area', cache['area'].astype(float)),
                             ('owners', np.ones(len(cache['area']),dtype=np.int64))]:
            bad={**cache,field:value}
            with self.assertRaises(ValueError):preparation._validate_inputs(maps,protocol,bad,labels,'valid')
        for split in ('train','smoke','Test'):
            with self.assertRaises(ValueError):preparation._validate_inputs(maps,protocol,cache,labels,split)
        for workers in (-1,True,1.5):
            with self.assertRaises(ValueError):
                preparation.prepare('unused',maps,protocol,cache,labels,'context',{},workers=workers)
        for ids in ([0,0],[4],[-1],[.5],[[0]], [True]):
            with self.assertRaises(ValueError):
                preparation.prepare('unused',maps,protocol,cache,labels,'context',{},audit_ids=ids)
        for binding in ({},{'relative':'hash'}):
            with self.assertRaises(ValueError):preparation._source_check(binding)


if __name__ == '__main__':
    unittest.main()
