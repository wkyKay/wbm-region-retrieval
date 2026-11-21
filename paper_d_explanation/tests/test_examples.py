import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper_d_explanation import examples, figures


class ExampleTests(unittest.TestCase):
    def test_relation_accepts_binary_float_and_distinguishes_extra_labels(self):
        q = np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=float)
        cases = [
            (q.copy(), 'exact', 1.),
            (np.array([1, 1, 0, 0, 0, 0, 1, 0]), 'full_extra', 1.),
            (np.array([1, 1, 0, 0, 0, 0, 0, 0]), 'partial', .5),
            (np.array([0, 1, 0, 0, 0, 0, 0, 0]), 'none', 0.),
        ]
        for c, kind, cov in cases:
            result = examples.relation(q, c)
            self.assertEqual((result['relation'], result['coverage']), (kind, cov))
        with self.assertRaises(ValueError):
            examples.relation(q, np.zeros(8))

    def test_representative_follows_frozen_order_not_smallest_id(self):
        classes = np.array([0, 0, 0, 1, 1, 2])
        result = examples.anchor_queries([2, 1, 4, 3, 5], classes)
        self.assertEqual(result, {0: 2, 1: 4, 2: 5})

    def test_display_categories_have_fixed_fallback_and_no_contribution_selection(self):
        def row(q, c, rank, stage, relation, score):
            return dict(query_id=q, candidate_id=c, rank=rank, stage=stage,
                        relation=relation, ranking_score=score)
        anchors = [row(8, 90, 1, 'test', 'exact', .99),
                   row(2, 70, 100, 'test', 'exact', .25),
                   row(2, 60, 10, 'test', 'exact', .5),
                   row(8, 92, 100, 'cross_test', 'partial', .2)]
        fallback = [row(1, 5, 1, 'test', 'exact', .99),
                    row(4, 80, 1, 'cross_test', 'full_extra', .8)]
        result = {r['category']: r for r in examples.choose_display(anchors, fallback)}
        self.assertEqual(result['ordinary_exact']['candidate_id'], 60)
        self.assertEqual(result['ordinary_exact']['selection_pool'], 'anchor_pool')
        self.assertEqual(result['cross_full']['selection_pool'], 'all_test_rank1_fallback')
        self.assertEqual(result['cross_partial']['candidate_id'], 92)
        self.assertEqual(result['cross_none']['status'], 'missing')
        anchors.append(dict(row(1, 9, 1, 'cross_test', 'none', .8), gallery_has_coverage=False))
        fallback.append(dict(row(5, 10, 1, 'cross_test', 'none', .7), gallery_has_coverage=True))
        result = {r['category']: r for r in examples.choose_display(anchors, fallback)}
        self.assertEqual(result['cross_none']['query_id'], 5)
        self.assertEqual(result['cross_none']['selection_pool'], 'all_test_rank1_fallback')

    def test_masks_bind_local_ids_areas_and_original_defect_pixels(self):
        raw = np.ones((8, 8), dtype=np.uint8)
        mask = np.zeros_like(raw)
        mask[1:3, 1:4] = 1
        mask[5:7, 5:7] = 2
        raw[mask > 0] = 2
        cache = dict(map_ids=np.array([27]), offsets=np.array([0, 2]), area=np.array([6, 4]))
        self.assertEqual(examples.validate_mask(27, raw, mask, cache, 0)['region_areas'], [6, 4])
        with self.assertRaises(ValueError):
            examples.validate_mask(28, raw, mask, cache, 0)
        wrong = mask.copy()
        wrong[0, 0] = 1
        with self.assertRaises(ValueError):
            examples.validate_mask(27, raw, wrong, cache, 0)
        swapped = mask.copy()
        swapped[mask == 1] = 2; swapped[mask == 2] = 1
        with self.assertRaises(ValueError):
            examples.validate_mask(27, raw, swapped, cache, 0)

    def test_contributions_use_complete_query_denominator_and_record_rounding(self):
        areas = np.array([9, 16])
        c, t, a = .81, .93, .7
        score = .75 * (float(np.float32(.4) * np.float32(c)) + .6 * t) + .25 * a
        contribution = 3 * score / 7
        e = dict(query_weight_denominator=7., score=contribution, matches=[dict(
            query_local_index=0, candidate_local_index=1, C=c, T=t, A=a,
            contribution=contribution)])
        r = dict(ranking_score=contribution, query_id=2, candidate_id=4)
        result = examples.enrich_explanation(e, r, areas)
        self.assertEqual(result['matches'][0]['query_normalized_weight'], 3 / 7)
        self.assertAlmostEqual(result['score_reconstruction_error'], 0.)
        self.assertGreater(result['algebraic_decomposition_error'], 0.)
        self.assertLess(result['algebraic_decomposition_error'], 2e-6)
        bad = copy.deepcopy(e)
        bad['query_weight_denominator'] = 3.
        with self.assertRaises(ValueError):
            examples.enrich_explanation(bad, r, areas)
        with self.assertRaises(ValueError):
            examples.enrich_explanation(e, dict(r, ranking_score=contribution + .01), areas)

    def test_unselected_source_defects_stay_gray(self):
        raw = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        mask = np.array([[0, 0], [1, 0]], dtype=np.uint8)
        rgb = figures.map_rgb(raw, mask, {0: '#0072B2'})
        np.testing.assert_allclose(rgb[1, 0], figures._hex_rgb('#0072B2'))
        np.testing.assert_allclose(rgb[1, 1], figures._hex_rgb('#A6ADB5'))
        self.assertFalse(np.array_equal(rgb[1, 1], rgb[0, 0]))


def synthetic_export(path):
    """Tiny deterministic original-pixel fixture for renderer QA, not evidence."""
    folder = Path(path) / 'explanations'
    folder.mkdir(parents=True)
    raw = np.zeros((52, 52), dtype=np.uint8)
    yy, xx = np.indices(raw.shape)
    raw[(xx - 25.5) ** 2 + (yy - 25.5) ** 2 < 24 ** 2] = 1
    mask = np.zeros_like(raw)
    mask[12:19, 14:23] = 1
    mask[30:38, 29:40] = 2
    raw[mask > 0] = 2
    raw[24, 8:12] = 2
    a = np.array([63, 88])
    denominator = float(np.sqrt(a).sum())
    rows, chosen = [], []
    for n, (cat, stage, kind) in enumerate(examples.DISPLAY_CATEGORIES):
        qlabels = [1, 0, 0, 0, 0, 0, 1, 0]
        clabels = {'exact': qlabels, 'full_extra': [1, 1, 0, 0, 0, 0, 1, 0],
                   'partial': [1, 0, 0, 0, 0, 0, 0, 0],
                   'none': [0, 0, 0, 0, 1, 0, 0, 0]}[kind]
        matches = []
        total = 0.
        for i, (c, t, ar) in enumerate(((.8, .95, .9), (.5, .9, .7))):
            v = .3 * c + .45 * t + .25 * ar
            cont = np.sqrt(a[i]) / denominator * v
            total += cont
            matches.append(dict(query_local_index=i, candidate_local_index=1 - i,
                C=c, T=t, A=ar, contribution=float(cont), raw_score=v))
        rec = dict(stage=stage, query_id=1, candidate_id=2, rank=(1 if n < 2 else 100),
                   ranking_score=float(total), anchor=True, **examples.relation(qlabels, clabels))
        e = examples.enrich_explanation(dict(query_weight_denominator=denominator,
            matches=matches, score=float(total)), rec, a)
        e['example_id'] = cat
        rows.append(e)
        chosen.append(dict(status='selected', category=cat, example_id=cat, selection_pool='synthetic'))
    np.savez_compressed(folder / 'assets.npz', sample_ids=np.array([1, 2]),
                        raw_maps=np.stack([raw, raw]), region_masks=np.stack([mask, mask]))
    (folder / 'examples.json').write_text(json.dumps(dict(examples=rows)))
    (folder / 'selection.json').write_text(json.dumps(dict(display_cases=chosen)))
    files = {p.name: examples._hash(p) for p in folder.iterdir()}
    (folder / 'verification.json').write_text(json.dumps(dict(status='verified', files=files)))


if __name__ == '__main__':
    unittest.main()
