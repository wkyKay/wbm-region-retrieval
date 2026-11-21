"""Publication figures from sealed real D1--D3 examples, without model access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np


COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
COMPONENT_COLORS = ('#4C78A8', '#E69F00', '#59A14F')
CATEGORY_TITLES = {
    'ordinary_exact': 'Ordinary retrieval · exact match',
    'cross_full': 'Cross-combination · full coverage',
    'cross_partial': 'Cross-combination · partial coverage',
    'cross_none': 'Cross-combination · no label overlap',
}


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 10, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    return plt


def _hex_rgb(value):
    return np.asarray([int(value[i:i + 2], 16) / 255. for i in (1, 3, 5)])


def map_rgb(raw, mask, region_colors):
    """Every source defect remains visible, including points outside Top-4."""
    raw, mask = np.asarray(raw), np.asarray(mask)
    rgb = np.ones((*raw.shape, 3), dtype=float)
    rgb[raw == 1] = _hex_rgb('#F0F3F6')
    rgb[raw == 3] = _hex_rgb('#E1E5E9')
    rgb[raw == 2] = _hex_rgb('#A6ADB5')
    for local, color in region_colors.items():
        rgb[mask == int(local) + 1] = _hex_rgb(color)
    return rgb


def _draw_map(ax, raw, mask, region_colors, prefix, *, compact=False):
    import matplotlib.patheffects as pe
    ax.imshow(map_rgb(raw, mask, region_colors), interpolation='nearest', origin='upper')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#DDE2E7'); spine.set_linewidth(.7)
    centers = {}
    height, width = raw.shape
    for local in range(int(mask.max())):
        ys, xs = np.where(mask == local + 1)
        if not len(xs):
            raise ValueError('Noncontiguous original region mask labels')
        center = (float(xs.mean()), float(ys.mean()))
        centers[local] = center
        ax.text(center[0], center[1], f'{prefix}{local + 1}', fontsize=7 if compact else 9,
                fontweight='bold', color='#17232F', ha='center', va='center', zorder=5,
                path_effects=[pe.withStroke(linewidth=2.7, foreground='white')])
    ax.set_xlim(-.5, width - .5); ax.set_ylim(height - .5, -.5)
    return centers


def _draw_contributions(ax, example, *, compact=False):
    n = example['query_regions']
    matches = {int(m['query_local_index']): m for m in example['matches']}
    vals = np.zeros((n, 3))
    labels = []
    for local in range(n):
        match = matches.get(local)
        weight = np.sqrt(example['query_region_areas'][local]) / example['query_weight_denominator']
        if match:
            vals[local] = [match['contribution_' + x] for x in ('C', 'T', 'A')]
            label = f"Q{local + 1} → R{match['candidate_local_index'] + 1}  (w={weight:.3f})"
            if not compact:
                label += f"\nC={match['C']:.3f}  T={match['T']:.3f}  A={match['A']:.3f}"
            labels.append(label)
        else:
            labels.append(f'Q{local + 1} → unmatched  (w={weight:.3f})')
    y = np.arange(n)
    left = np.zeros(n)
    for col, (color, name) in enumerate(zip(COMPONENT_COLORS, ('CNN C', 'Structure T', 'Area A'))):
        ax.barh(y, vals[:, col], left=left, color=color, height=.44, label=name)
        left += vals[:, col]
    bound = max(float(left.max(initial=0)) * 1.2, .15)
    for i, total in enumerate(left):
        ax.text(total + bound * .025, i, f'{total:.3f}', va='center', fontsize=8 if compact else 10)
    ax.set_xlim(0, bound)
    ax.set_ylim(n - .5, -.7)
    ax.set_yticks(y, labels, fontsize=7 if compact else 8)
    ax.set_xlabel('Contribution to candidate score', fontsize=8 if compact else 10)
    ax.xaxis.grid(True, color='#E9EDF0', linewidth=.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#CDD3DA')
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=7 if compact else 9)
    ax.set_title(f"Actual score = {example['ranking_score']:.5f}", loc='left',
                 fontsize=9 if compact else 11, fontweight='bold', pad=12)
    if not compact:
        ax.legend(loc='lower left', bbox_to_anchor=(-.015, -.43), ncol=3, frameon=False,
                  handlelength=1., columnspacing=1.0, fontsize=8.5)


def _label_text(names, width=28):
    return '\n'.join(textwrap.wrap(' + '.join(names), width=width, break_long_words=False))


def _draw_case(fig, axes, example, assets, title, *, compact=False):
    from matplotlib.patches import ConnectionPatch
    qax, cax, bax = axes
    q, c = example['query_id'], example['candidate_id']
    qraw, qmask = assets[q]
    craw, cmask = assets[c]
    qcolors = {i: COLORS[i % len(COLORS)] for i in range(example['query_regions'])}
    ccolors = {int(m['candidate_local_index']): qcolors[int(m['query_local_index'])]
               for m in example['matches']}
    # Candidate regions that did not contribute remain dark gray but numbered.
    ccolors.update({i: '#68737D' for i in range(int(cmask.max())) if i not in ccolors})
    qp = _draw_map(qax, qraw, qmask, qcolors, 'Q', compact=compact)
    cp = _draw_map(cax, craw, cmask, ccolors, 'R', compact=compact)
    for match in example['matches']:
        qi, ci = int(match['query_local_index']), int(match['candidate_local_index'])
        link = ConnectionPatch(xyA=qp[qi], coordsA=qax.transData, xyB=cp[ci], coordsB=cax.transData,
            color=qcolors[qi], alpha=.38, linewidth=.8 if compact else 1.2,
            linestyle='-', zorder=1, clip_on=False)
        fig.add_artist(link)
    size = 8 if compact else 10
    qax.set_title(f'Query {q}', fontsize=size, fontweight='bold', pad=7)
    cax.set_title(f'Candidate {c} · rank {example["rank"]}', fontsize=size, fontweight='bold', pad=7)
    qax.set_xlabel(_label_text(example['query_label_names']), fontsize=7 if compact else 9, labelpad=7)
    cax.set_xlabel(_label_text(example['candidate_label_names']), fontsize=7 if compact else 9, labelpad=7)
    _draw_contributions(bax, example, compact=compact)
    # Place each panel heading above the map pair, without overlapping labels.
    box = qax.get_position()
    fig.text(box.x0, box.y1 + ((.40 if compact else .50) / fig.get_figheight()), title,
             fontsize=9 if compact else 12, fontweight='bold', color='#203348')


def _save(fig, stem, *, png=True, pdf=True, dpi=220):
    paths = []
    if png:
        path = Path(str(stem) + '.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.16)
        paths.append(path)
    if pdf:
        path = Path(str(stem) + '.pdf')
        fig.savefig(path, bbox_inches='tight', pad_inches=.16)
        paths.append(path)
    return paths


def _load(output):
    folder = Path(output) / 'explanations'
    verification = json.loads((folder / 'verification.json').read_text())
    if verification.get('status') != 'verified':
        raise ValueError('Only verified explanation data can be visualized')
    for name, sha in verification['files'].items():
        if _hash(folder / name) != sha:
            raise ValueError('Explanation artifact changed: ' + name)
    payload = json.loads((folder / 'examples.json').read_text())
    selection = json.loads((folder / 'selection.json').read_text())
    with np.load(folder / 'assets.npz', allow_pickle=False) as archive:
        assets = {int(s): (archive['raw_maps'][i].copy(), archive['region_masks'][i].copy())
                  for i, s in enumerate(archive['sample_ids'])}
    return payload, selection, assets


def render(output, *, gallery=True):
    """Create the main panel, individual cases and all-anchor appendix gallery."""
    plt = _pyplot()
    output = Path(output)
    payload, selection, assets = _load(output)
    by_id = {e['example_id']: e for e in payload['examples']}
    cases = [x for x in selection['display_cases'] if x['status'] == 'selected']
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    existing = list(folder.glob('main_cases.*')) + list(folder.glob('figures_manifest.json'))
    if existing:
        raise FileExistsError('Figures already exist; preserve the original render')
    produced = []
    # The panel uses generous row spacing so masks, label combinations and score
    # bars remain legible when the image is inspected or exported at full size.
    fig = plt.figure(figsize=(14.8, 4.5 * len(cases)))
    outer = fig.add_gridspec(len(cases), 3, width_ratios=(1.0, 1.0, 1.85),
                            left=.035, right=.97, top=.90, bottom=.12,
                            hspace=.85, wspace=.62)
    for i, case in enumerate(cases):
        axes = [fig.add_subplot(outer[i, j]) for j in range(3)]
        _draw_case(fig, axes, by_id[case['example_id']], assets,
                   f'({chr(97 + i)}) {CATEGORY_TITLES[case["category"]]}')
    fig.suptitle('Region-level evidence in actual test retrieval', fontsize=17,
                 fontweight='bold', y=.992, x=.035, ha='left', color='#172B3A')
    fig.text(.035, .974, 'Frozen regional J + F · original masks · greedy one-to-one matching',
             fontsize=11, color='#566473')
    fig.text(.035, .015,
        'Colored pixels: selected regions. Gray defects: outside Top-4 or unmatched candidate regions.\n'
        'Lines show actual accepted correspondences; wafer-level labels do not assign semantic labels to individual regions.\n'
        'Bars show weighted 0.30C / 0.45T / 0.25A terms; numerical rounding residuals are recorded in the source data.',
        fontsize=9, color='#566473', va='bottom')
    produced += _save(fig, folder / 'main_cases')
    plt.close(fig)
    individual = folder / 'individual'
    individual.mkdir(exist_ok=True)
    for case in cases:
        fig = plt.figure(figsize=(14.8, 4.8))
        grid = fig.add_gridspec(1, 3, width_ratios=(1., 1., 1.85),
                              left=.035, right=.97, top=.76, bottom=.25, wspace=.62)
        axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
        _draw_case(fig, axes, by_id[case['example_id']], assets, CATEGORY_TITLES[case['category']])
        fig.text(.035, .96, 'Actual test retrieval · region correspondences and score contributions',
                 fontsize=14, fontweight='bold', va='top')
        fig.text(.035, .035, 'Gray defects remain visible. Region numbers are geometric components, not defect-type annotations.',
                 fontsize=9, color='#566473')
        produced += _save(fig, individual / case['category'])
        plt.close(fig)
    gallery_count, gallery_pages = 0, 0
    if gallery:
        from matplotlib.backends.backend_pdf import PdfPages
        preview = folder / 'gallery'
        preview.mkdir(exist_ok=True)
        path = preview / 'all_222_anchor_examples.pdf'
        anchors = [e for e in payload['examples'] if e['anchor']]
        with PdfPages(path) as pdf:
            for start in range(0, len(anchors), 6):
                group = anchors[start:start + 6]
                fig = plt.figure(figsize=(14.8, 20.8))
                grid = fig.add_gridspec(len(group), 3, width_ratios=(1., 1., 1.85),
                    left=.035, right=.97, top=.955, bottom=.05, hspace=1.05, wspace=.62)
                for i, example in enumerate(group):
                    axes = [fig.add_subplot(grid[i, j]) for j in range(3)]
                    _draw_case(fig, axes, example, assets,
                        f'{start + i + 1:03d} · {example["stage"]} · {example["relation"]} · coverage {example["coverage"]:.2f}',
                        compact=True)
                fig.suptitle(f'Fixed test example gallery · page {start // 6 + 1}',
                             fontsize=15, x=.035, ha='left', y=.992)
                fig.text(.035, .012, 'Same frozen method and masks in every panel. All 37 class representatives × ranks 1, 10, 100 × two protocols.',
                         fontsize=8, color='#566473')
                pdf.savefig(fig, bbox_inches='tight', pad_inches=.16)
                produced += _save(fig, preview / f'page_{start // 6 + 1:02d}', pdf=False, dpi=130)
                plt.close(fig)
                gallery_count += len(group); gallery_pages += 1
        produced.append(path)
    caption_rows = []
    for case in cases:
        e = by_id[case['example_id']]
        caption_rows.append({'category': case['category'], 'example_id': e['example_id'],
            'stage': e['stage'], 'query_id': e['query_id'], 'candidate_id': e['candidate_id'],
            'rank': e['rank'], 'score': e['ranking_score'], 'relation': e['relation'],
            'coverage': e['coverage'], 'selection_pool': case['selection_pool'],
            'query_labels': ' + '.join(e['query_label_names']),
            'candidate_labels': ' + '.join(e['candidate_label_names'])})
    with (folder / 'display_cases.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(caption_rows[0]))
        writer.writeheader(); writer.writerows(caption_rows)
    produced.append(folder / 'display_cases.csv')
    from PIL import Image
    dimensions = {}
    for image in produced:
        if image.suffix == '.png':
            with Image.open(image) as bitmap:
                dimensions[str(image.relative_to(output))] = list(bitmap.size)
    result = {'status': 'rendered', 'main_display_cases': len(cases),
        'main_figure': 'figures/main_cases.png', 'main_vector_figure': 'figures/main_cases.pdf',
        'anchor_gallery_examples': gallery_count, 'anchor_gallery_pages': gallery_pages,
        'source_explanation_verification_sha256': _hash(output / 'explanations/verification.json'),
        'renderer_source_sha256': _hash(Path(__file__)), 'png_dimensions': dimensions,
        'missing_categories': [x['category'] for x in selection['display_cases'] if x['status'] != 'selected'],
        'files': {str(p.relative_to(output)): _hash(p) for p in produced},
        'visual_qa': 'pending independent inspection; rendered is not an inspection claim'}
    with (folder / 'figures_manifest.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--no-gallery', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render(args.output, gallery=not args.no_gallery), indent=2))
