"""Fixed train-only exposure audit and unselected augmentation examples."""
from pathlib import Path

import cv2
import numpy as np

from .augmentation import AUGMENTATION_SPECS, PaperViews, VARIANTS
from .state import ROOT, PLAN, record, run_context, seal, signed, verified
from wbm_retrieval.common import digest, file_hash, read_json, write_csv

COHORT_SIZE = 1024
SEED = 2026
EPOCH = 0
EXAMPLE_COUNT = 8
REQUIRED = {'manifest.json', 'cohort.csv', 'per_token.csv', 'summary.csv',
            'summary.json', 'examples.png', 'examples.svg', 'verification.json'}


def select_tokens(population, count=COHORT_SIZE, seed=SEED):
    """Uniform sampling without labels; retain draw order for the eight examples."""
    if not isinstance(population, (int, np.integer)) or not 0 < count <= population:
        raise ValueError('Audit cohort must be a nonempty subset of train tokens')
    return np.random.default_rng(seed).choice(int(population), size=count, replace=False)


def build_cohort(data, *, count=COHORT_SIZE, seed=SEED):
    patches = data['patches']
    maps, regions = np.asarray(data['token_map_ids']), np.asarray(data['token_region_ids'])
    if (patches.shape[1:] != (3, 96, 96) or maps.shape != (len(patches),)
            or regions.shape != maps.shape or maps.dtype.kind not in 'iu'
            or regions.dtype.kind not in 'iu' or np.any(maps < 0) or np.any(regions < 1)):
        raise ValueError('Train patch/token identity arrays disagree')
    tokens = select_tokens(len(patches), count, seed)
    return [{'selection_rank': rank, 'token_index': int(token),
             'sample_id': int(maps[token]), 'map_id': int(maps[token]),
             'region_id': int(regions[token])} for rank, token in enumerate(tokens)]


def summarize_rows(rows):
    """Explicit eligible-token and source-pixel denominators, never zero-fill NA."""
    if not rows or len({row['model'] for row in rows}) != 1:
        raise ValueError('Summaries require one nonempty augmentation cohort')
    if len({row['token_index'] for row in rows}) != len(rows):
        raise ValueError('Repeated token would change the audit weighting')
    n = len(rows)
    result = {'model': rows[0]['model'], 'num_tokens': n}
    for flag in ('applied', 'no_op', 'empty_target_before', 'empty_target_after'):
        values = [int(row[flag]) for row in rows]
        if any(value not in (0, 1) for value in values):
            raise ValueError('Audit flags must be binary')
        result[flag+'_count'] = sum(values)
        result[flag+'_fraction'] = sum(values)/n
    for prefix in ('target', 'support'):
        counts = [(int(row[prefix+'_before_pixels']), int(row[prefix+'_retained_source_pixels']))
                  for row in rows]
        if any(before < 0 or not 0 <= retained <= before for before, retained in counts):
            raise ValueError('Invalid source-space pixel retention')
        eligible = [(before, retained) for before, retained in counts if before > 0]
        before_total = sum(before for before, _ in counts)
        retained_total = sum(retained for _, retained in counts)
        result[prefix+'_retention_eligible'] = len(eligible)
        result[prefix+'_before_pixels'] = before_total
        result[prefix+'_after_pixels'] = sum(int(row[prefix+'_after_pixels']) for row in rows)
        result[prefix+'_retained_source_pixels'] = retained_total
        result[prefix+'_retention_fraction_mean'] = (
            float(np.mean([retained/before for before, retained in eligible])) if eligible else None)
        result[prefix+'_retention_fraction_pixel_weighted'] = (
            retained_total/before_total if before_total else None)
    newly_empty = sum(int(row['target_before_pixels'] > 0 and row['target_after_pixels'] == 0)
                      for row in rows)
    if newly_empty != sum(int(row['newly_empty_target']) for row in rows):
        raise ValueError('Newly empty target flags disagree with pixel counts')
    result['newly_empty_target_count'] = newly_empty
    eligible = result['target_retention_eligible']
    result['newly_empty_target_fraction'] = newly_empty/eligible if eligible else None
    for name in ('changed_positions', 'noise_selected_pixels', 'noise_changed_pixels',
                 'noise_added_defects', 'noise_removed_defects'):
        result[name] = sum(int(row[name]) for row in rows)
    if (result['noise_changed_pixels'] != result['noise_added_defects']+result['noise_removed_defects']
            or not 0 <= result['noise_changed_pixels'] <= result['noise_selected_pixels']
            <= result['support_before_pixels']):
        raise ValueError('Noise selection/change counters disagree')
    result['changed_position_fraction'] = result['changed_positions']/(n*96*96)
    for key in ('noise_selected', 'noise_changed'):
        result[key+'_fraction_of_original_support'] = (
            result[key+'_pixels']/result['support_before_pixels']
            if result['support_before_pixels'] else None)
    return result


def _source_binding(data):
    cache = Path(data['cache']).resolve()
    marker = data['prepared_marker']
    bound = {}
    for suffix in ('F.npy', 'regions.npz', 'region_labels.npy'):
        path = cache/('train_'+suffix)
        actual = file_hash(path)
        if actual != marker['files'].get('cache/'+path.name):
            raise ValueError('Frozen train audit source changed: '+str(path))
        bound[str(path)] = actual
    path = cache/'train_prepared.json'
    if read_json(path) != marker:
        raise ValueError('In-memory train prepared marker differs from its source')
    bound[str(path)] = file_hash(path)
    return bound


def _overlay(patch):
    """Wafer support gray, other defects orange, target evidence blue, void white."""
    patch = np.asarray(patch)
    rgb = np.full((96, 96, 3), 255, dtype=np.uint8)
    rgb[patch[2] > 0] = (224, 228, 235)
    rgb[(patch[0] > 0) & (patch[1] == 0)] = (220, 105, 45)
    rgb[patch[1] > 0] = (20, 108, 186)
    return rgb


def _draw_examples(folder, examples, signature):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    titles = ['Original', 'Crop', 'Cutout', 'Noise', 'Rotation', 'Shift']
    with plt.rc_context({'svg.hashsalt': signature, 'font.family': 'DejaVu Sans'}):
        figure, axes = plt.subplots(len(examples), 6, figsize=(12, 2.1*len(examples)+.8),
                                    squeeze=False, layout='constrained')
        for r, example in enumerate(examples):
            original = example['original']
            for c, variant in enumerate((None, *VARIANTS)):
                axis = axes[r, c]
                patch = original if variant is None else example['views'][variant]
                axis.imshow(_overlay(patch), interpolation='nearest')
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    axis.set_title(titles[c], fontsize=11)
                if variant is None:
                    info = example['identity']
                    label = f"sample {info['sample_id']}; region {info['region_id']}\ntoken {info['token_index']}"
                else:
                    stats = example['stats'][variant]
                    parts = []
                    for prefix, abbreviation in (('target', 'T'), ('support', 'W')):
                        value = stats[prefix+'_retention_fraction']
                        parts.append(abbreviation+(' n/a' if value is None else f' {value:.1%}'))
                    label = '; '.join(parts)
                    if stats['empty_target_after']:
                        label += '\nempty target'
                axis.set_xlabel(label, fontsize=8)
        figure.suptitle('Fixed train sample: first 8 of 1024 uniform token draws; epoch 0\n'
                        'Original input and five separate augmentations (no example selection by appearance)',
                        fontsize=12)
        legend = [Patch(color=np.array(color)/255, label=label) for color, label in
                  [((224, 228, 235), 'Wafer support'), ((220, 105, 45), 'Other defects'),
                   ((20, 108, 186), 'Target evidence')]]
        figure.legend(handles=legend, loc='outside lower center', ncol=3, frameon=False,
                      title='T / W: unique original 96x96 input pixels retained; not output area or raw die counts')
        figure.savefig(folder/'examples.png', dpi=180, metadata={'Title': 'Fixed train augmentation audit'})
        figure.savefig(folder/'examples.svg', metadata={'Date': None, 'Creator': 'cnn_augmentation.audit'})
        plt.close(figure)


def run_audit(output, inputs):
    """Seal the fixed label-free train cohort, exposure tables and example figures."""
    import matplotlib
    output = Path(output).resolve()
    context = run_context(output)
    if (inputs.get('refinement_context_signature') != context['signature']
            or inputs.get('plan_signature') != context['plan_signature']
            or inputs['config']['seed'] != SEED):
        raise ValueError('Augmentation audit is detached from the frozen run/seed')
    data = inputs['datasets']['train']
    cohort = build_cohort(data)
    binding = _source_binding(data)
    code_names = ['cnn_augmentation/audit.py', 'cnn_augmentation/augmentation.py',
                  'cnn_augmentation/state.py', 'experiments/proposed/_shared/ef_suite.py',
                  'f_region_structure/state.py', 'wbm_retrieval/common.py']
    science = {'schema_version': 1, 'kind': 'fixed_train_augmentation_audit',
               'context_signature': context['signature'], 'plan_signature': context['plan_signature'],
               'scope': 'train only, epoch0; labels are not read for selection or audit',
               'labels_used': False, 'population_tokens': len(data['patches']),
               'cohort_size': COHORT_SIZE, 'seed': SEED, 'epoch': EPOCH,
               'selection': 'numpy default_rng(seed).choice(population,1024,replace=False), draw order retained',
               'cohort': cohort, 'cohort_sha256': digest(cohort),
               'example_tokens': [row['token_index'] for row in cohort[:EXAMPLE_COUNT]],
               'examples_selection': 'first eight draws; no label, appearance or result filtering',
               'data_binding': binding, 'augmentation_specs': AUGMENTATION_SPECS,
               'plan_augmentation_specs': PLAN['single_augmentations'],
               'code': {name: file_hash(ROOT/name) for name in code_names},
               'versions': {'numpy': np.__version__, 'opencv': cv2.__version__,
                            'matplotlib': matplotlib.__version__},
               'retention_definition': 'unique original96 source pixel IDs actually sampled into augmented '
                                       'target/support, divided by original target/support pixels; '
                                       'no target-aware rejection or resampling'}
    manifest = signed(science)
    signature = manifest['signature']
    folder = output/'augmentation_audit'
    if verified(folder, signature):
        marker = read_json(folder/'completed.json')
        actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
        if set(marker['files']) != REQUIRED or actual != REQUIRED | {'completed.json'}:
            raise ValueError('Augmentation audit has incomplete or unexpected sealed artifacts')
        summary = read_json(folder/'summary.json')
        if (summary.get('signature') != signature or summary.get('context_signature') != context['signature']
                or set(summary.get('models', {})) != set(VARIANTS)):
            raise ValueError('Augmentation audit summary identity changed')
        return summary
    folder.mkdir(parents=True, exist_ok=True)
    record(folder/'manifest.json', manifest)
    write_csv(folder/'cohort.csv', cohort, list(cohort[0]))
    examples = [{'identity': row, 'original': np.array(data['patches'][row['token_index']], copy=True),
                 'views': {}, 'stats': {}} for row in cohort[:EXAMPLE_COUNT]]
    examples_by_token = {row['identity']['token_index']: row for row in examples}
    rows, models = [], {}
    for variant in VARIANTS:
        dataset = PaperViews(data['cache'], 'train', inputs['spec'], SEED, variant, epoch=EPOCH)
        if len(dataset) != len(data['patches']):
            raise ValueError('PaperViews changed the original train token pool')
        model_rows = []
        for identity in cohort:
            index, original, view, stats = dataset[identity['token_index']]
            original, view = original.numpy(), view.numpy()
            if index != identity['token_index'] or not np.array_equal(original, data['patches'][index]):
                raise ValueError('Augmentation audit changed first-view identity or pixels')
            if stats['no_op'] != int(np.array_equal(original, view)):
                raise ValueError('Augmentation no-op accounting is inconsistent')
            stats = dict(stats)
            for prefix in ('target', 'support'):
                before = int(original[1 if prefix == 'target' else 2].sum())
                if before != stats[prefix+'_before_pixels']:
                    raise ValueError('Original audit pixel count differs from source input')
                if before == 0:
                    stats[prefix+'_retention_fraction'] = None
            row = {'model': variant, **identity, **stats}
            model_rows.append(row)
            if index in examples_by_token:
                examples_by_token[index]['views'][variant] = view.copy()
                examples_by_token[index]['stats'][variant] = stats
        models[variant] = summarize_rows(model_rows)
        rows.extend(model_rows)
    write_csv(folder/'per_token.csv', rows, list(rows[0]))
    write_csv(folder/'summary.csv', list(models.values()), list(next(iter(models.values()))))
    _draw_examples(folder, examples, signature)
    if any(file_hash(path) != expected for path, expected in binding.items()):
        raise ValueError('Train audit modified a protected input')
    verification = {'status': 'verified', 'signature': signature,
                    'source_inputs_unchanged': True, 'labels_used': False,
                    'shared_cohort_all_variants': True, 'first_views_match_frozen_cache': True,
                    'no_target_aware_resampling': True, 'source_id_retention': True,
                    'tokens_per_variant': COHORT_SIZE, 'rows': len(rows),
                    'figure_visual_review': 'not yet reviewed; PNG/SVG are deterministic audit outputs'}
    summary = {'signature': signature, 'context_signature': context['signature'],
               'scope': science['scope'], 'labels_used': False,
               'cohort': {key: science[key] for key in ('population_tokens', 'cohort_size', 'seed', 'epoch',
                                                     'cohort_sha256', 'example_tokens', 'selection')},
               'models': models, 'figures': ['examples.png', 'examples.svg'],
               'denominators': {'token_fractions': 'all1024 unless explicitly eligible',
                                'retention_fraction_mean': 'unweighted mean over original-nonempty tokens',
                                'retention_fraction_pixel_weighted': 'sum retained source pixels / sum original pixels',
                                'newly_empty_target_fraction': 'original-nonempty target tokens',
                                'noise_fractions': 'all original support pixels in the fixed cohort'},
               'limits': ['Descriptive fixed epoch0 train sample, not the all-epoch training exposure.',
                          'Original empty targets/support have undefined retention (null), not zero.',
                          'Source pixels refer to cached96 input pixels, not original raw die counts.',
                          'Examples are the first eight draws without visual or outcome selection.']}
    record(folder/'verification.json', verification)
    record(folder/'summary.json', summary)
    if {str(p.relative_to(folder)) for p in folder.iterdir() if p.is_file()} != REQUIRED:
        raise ValueError('Unexpected files in the isolated augmentation audit')
    seal(folder, signature)
    verified(folder, signature)
    return summary
