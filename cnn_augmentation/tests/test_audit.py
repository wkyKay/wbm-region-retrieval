"""Audit weighting, label exclusion, deterministic examples and immutable seals."""
from pathlib import Path

import numpy as np
import pytest

from cnn_augmentation.audit import (COHORT_SIZE, REQUIRED, _overlay, build_cohort,
                                    run_audit, select_tokens, summarize_rows)
from cnn_augmentation.state import PLAN, VERSION, signed
from wbm_retrieval.common import file_hash, read_json, write_json


class LabelTrap(dict):
    def __getitem__(self, key):
        if 'label' in key:
            raise AssertionError('Class labels must not be read by the augmentation audit')
        return super().__getitem__(key)


def _row(token, before, retained, after, *, support=100, support_retained=90, model='crop'):
    return {'model': model, 'token_index': token, 'applied': 1, 'no_op': 0,
            'target_before_pixels': before, 'target_retained_source_pixels': retained,
            'target_after_pixels': after, 'empty_target_before': int(before == 0),
            'empty_target_after': int(after == 0),
            'newly_empty_target': int(before > 0 and after == 0),
            'support_before_pixels': support, 'support_retained_source_pixels': support_retained,
            'support_after_pixels': 200 if support else 0, 'changed_positions': 10,
            'noise_selected_pixels': 0, 'noise_changed_pixels': 0,
            'noise_added_defects': 0, 'noise_removed_defects': 0}


def test_uniform_selection_preserves_rng_and_original_ids_without_labels():
    np.random.seed(57)
    state = np.random.get_state()
    selected = select_tokens(5000)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert len(selected) == COHORT_SIZE and len(set(selected)) == COHORT_SIZE
    assert np.array_equal(selected, select_tokens(5000))
    assert not np.array_equal(selected, select_tokens(5000, seed=2027))
    # First eight retain draw order, rather than choosing low IDs or pretty examples.
    assert not np.array_equal(selected, np.sort(selected))
    data = LabelTrap(patches=np.zeros((40, 3, 96, 96), dtype=np.uint8),
                     token_map_ids=np.arange(40, dtype=np.int64)*17+1000,
                     token_region_ids=np.arange(40, dtype=np.int64)%4+1,
                     labels=object(), singleton_labels=object())
    cohort = build_cohort(data, count=16)
    for rank, row in enumerate(cohort):
        token = row['token_index']
        assert row == {'selection_rank': rank, 'token_index': token, 'sample_id': token*17+1000,
                       'map_id': token*17+1000, 'region_id': token%4+1}
    assert cohort == build_cohort(data, count=16)


@pytest.mark.parametrize('population,count', [(0, 1), (10, 11), (10, 0)])
def test_invalid_selection_cannot_shrink_or_repeat_tokens(population, count):
    with pytest.raises(ValueError):
        select_tokens(population, count)


def test_target_retention_uses_source_pixels_and_eligible_denominators():
    rows = [_row(0, 4, 4, 16), _row(1, 100, 10, 40), _row(2, 0, 0, 0)]
    summary = summarize_rows(rows)
    assert summary['target_retention_eligible'] == 2
    assert summary['target_retention_fraction_mean'] == pytest.approx(.55)
    assert summary['target_retention_fraction_pixel_weighted'] == pytest.approx(14/104)
    assert summary['target_after_pixels'] == 56  # Deliberately different from retained source area.
    assert summary['empty_target_after_fraction'] == pytest.approx(1/3)
    assert summary['newly_empty_target_fraction'] == 0
    rows[1] = _row(1, 100, 0, 0)
    summary = summarize_rows(rows)
    assert summary['newly_empty_target_count'] == 1
    assert summary['newly_empty_target_fraction'] == .5
    assert summary['empty_target_after_fraction'] == pytest.approx(2/3)


def test_undefined_retention_is_null_and_noop_is_separate_from_applied():
    rows = [_row(0, 0, 0, 0, support=0, support_retained=0),
            _row(1, 0, 0, 0, support=0, support_retained=0)]
    rows[0].update(applied=0, no_op=1, changed_positions=0)
    rows[1].update(applied=1, no_op=1, changed_positions=0)
    result = summarize_rows(rows)
    assert result['applied_fraction'] == .5 and result['no_op_fraction'] == 1
    assert result['empty_target_after_fraction'] == 1
    assert result['newly_empty_target_fraction'] is None
    for field in ('target', 'support'):
        assert result[field+'_retention_fraction_mean'] is None
        assert result[field+'_retention_fraction_pixel_weighted'] is None
    assert result['noise_selected_fraction_of_original_support'] is None


def test_noise_rates_use_original_support_not_target_or_selected_pixels():
    rows = [_row(0, 20, 19, 19, support=100, model='noise'),
            _row(1, 20, 19, 19, support=300, model='noise')]
    for row in rows:
        row.update(noise_selected_pixels=10, noise_added_defects=3,
                   noise_removed_defects=1, noise_changed_pixels=4)
    summary = summarize_rows(rows)
    assert summary['noise_selected_fraction_of_original_support'] == .05
    assert summary['noise_changed_fraction_of_original_support'] == .02
    assert summary['noise_changed_pixels'] == 8
    rows[0]['noise_changed_pixels'] = 9
    with pytest.raises(ValueError, match='counters'):
        summarize_rows(rows)


def test_summary_rejects_duplicate_tokens_and_impossible_source_retention():
    row = _row(0, 4, 4, 16)
    with pytest.raises(ValueError, match='Repeated'):
        summarize_rows([row, row])
    with pytest.raises(ValueError, match='source-space'):
        summarize_rows([{**row, 'target_retained_source_pixels': 5}])


def test_overlay_keeps_three_channel_meaning_visible():
    patch = np.zeros((3, 96, 96), dtype=np.uint8)
    patch[2, 0, :3] = 1
    patch[0, 0, :2] = 1
    patch[1, 0, 0] = 1
    before = patch.copy()
    rgb = _overlay(patch)
    assert len({tuple(rgb[0, i]) for i in range(4)}) == 4
    assert np.array_equal(before, patch)


def _inputs_and_context(tmp_path):
    cache = tmp_path/'source/cache'
    cache.mkdir(parents=True)
    patches = np.zeros((COHORT_SIZE, 3, 96, 96), dtype=np.uint8)
    patches[:, 2, 8:88, 8:88] = 1
    patches[:, 0, 40:56, 40:56] = 1
    patches[:, 1, 42:50, 42:50] = 1
    patches[:, 0, 12:16, 20:30] = 1
    patches[::5, 1] = 0
    np.save(cache/'train_F.npy', patches)
    maps = np.arange(COHORT_SIZE, dtype=np.int64)*13+20
    regions = np.ones(COHORT_SIZE, dtype=np.int64)
    np.savez(cache/'train_regions.npz', map_ids=maps, offsets=np.arange(COHORT_SIZE+1),
             area=np.ones(COHORT_SIZE, dtype=np.int64))
    np.save(cache/'train_region_labels.npy', np.ones((COHORT_SIZE, 1, 1), dtype=np.uint8))
    marker = {'signature': 'source-fixture', 'files': {
        'cache/'+path.name: file_hash(path) for path in cache.iterdir()}}
    write_json(cache/'train_prepared.json', marker)
    output = tmp_path/'run'
    plan = signed({'plan': PLAN})
    context = signed({'version': VERSION, 'plan_signature': plan['signature']})
    write_json(output/'frozen_plan.json', plan)
    write_json(output/'run_manifest.json', context)
    data = LabelTrap(cache=cache, patches=np.load(cache/'train_F.npy', mmap_mode='r'),
                     token_map_ids=maps, token_region_ids=regions, prepared_marker=marker,
                     labels=object(), singleton_labels=object())
    inputs = LabelTrap(config={'seed': 2026}, spec={'preprocess': 'fixed_window', 'scale_range': None},
                       refinement_context_signature=context['signature'], plan_signature=plan['signature'],
                       datasets={'train': data}, labels=object())
    return output, inputs


def test_complete_audit_is_label_free_sealed_reusable_and_detects_tampering(tmp_path):
    output, inputs = _inputs_and_context(tmp_path)
    summary = run_audit(output, inputs)
    folder = output/'augmentation_audit'
    assert set(read_json(folder/'completed.json')['files']) == REQUIRED
    assert summary['labels_used'] is False
    assert summary['cohort']['cohort_size'] == COHORT_SIZE
    assert summary['cohort']['example_tokens'] == list(select_tokens(COHORT_SIZE)[:8])
    assert all(row['num_tokens'] == COHORT_SIZE for row in summary['models'].values())
    assert summary['models']['cutout']['applied_fraction'] < 1
    assert all(summary['models'][v]['applied_fraction'] == 1 for v in ('crop', 'noise', 'rotation', 'shift'))
    assert run_audit(output, inputs) == summary
    (folder/'unexpected.txt').write_text('unbound data')
    with pytest.raises(ValueError, match='unexpected'):
        run_audit(output, inputs)
    (folder/'unexpected.txt').unlink()
    (folder/'summary.csv').write_text('tampered')
    with pytest.raises(ValueError, match='Changed or missing'):
        run_audit(output, inputs)
