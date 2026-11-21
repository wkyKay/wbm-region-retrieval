"""Copy selected publication evidence without moving or re-evaluating sources.

The selection lists workspace-relative sources and exact byte hashes. Original
completion markers describe their original runs, not this partial export.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a safe relative path: ' + str(relative))
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes root: ' + str(path))
    return path


def write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError('Refuse to overwrite different content: ' + str(path))
    else:
        path.write_bytes(payload)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def csv_bytes(fields, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8')


def metric_count_rows(runs):
    """Extract each catalogue metric's actual aggregation denominator.

    Input: selected snapshot['runs'] with embedded source report data.
    Output: CSV-ready dictionaries. Missing counts remain '', never zero-filled.
    class_macro rows use effective_class_count; their effective_query_count is
    deliberately blank because query count is not the averaging denominator.
    num_queries/num_classes on those rows describe the underlying full report.
    """
    def count(value, context):
        if value is None:
            return ''
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Invalid count in source report: ' + context)
        return value

    def metric_bodies(data):
        yield 'all', data, False
        if 'class_macro_metrics' in data:
            yield 'class_macro', data, True

        def walk(value, prefix):
            if not isinstance(value, dict):
                return
            if 'metrics' in value:
                yield prefix, value, False
            else:
                for name, child in sorted(value.items()):
                    yield from walk(child, prefix + '/' + name if prefix else name)
        yield from walk(data.get('groups', {}), '')

    rows, seen = [], set()
    for run in runs:
        for family, report in run['reports'].items():
            data = report['data']
            version = family + '_v1'
            if data.get('metric_version') != version:
                raise ValueError('Report metric version differs from catalogue: ' + run['key'])
            for group, body, macro in metric_bodies(data):
                metrics = body['class_macro_metrics'] if macro else body['metrics']
                counts = body.get('class_macro_metric_class_counts' if macro else
                                  'metric_query_counts', {})
                if not isinstance(counts, dict):
                    raise ValueError('Metric counts must be a mapping: ' + run['key'])
                nq = count(body.get('num_queries'), run['key'] + '/' + group + '/num_queries')
                nc = count(body.get('num_classes'), run['key'] + '/' + group + '/num_classes')
                for name in sorted(metrics):
                    metric, k = name.rsplit('@', 1)
                    identity = (run['key'], version, group, metric, k)
                    if identity in seen:
                        raise ValueError('Duplicate metric count identity: ' + str(identity))
                    seen.add(identity)
                    n = count(counts.get(name), run['key'] + '/' + group + '/' + name)
                    upper = nc if macro else nq
                    if n != '' and upper != '' and n > upper:
                        raise ValueError('Effective count exceeds group population: ' + str(identity))
                    rows.append({
                        'run_id': run['run_id'],
                        'key': run['key'],
                        'repository': run['repository'],
                        'method': run['method'],
                        'variant': run['variant'],
                        'split': run['split'],
                        'protocol': run['protocol'],
                        'metric_version': version,
                        'group': group,
                        'metric': metric,
                        'K': k,
                        'num_queries': nq,
                        'num_classes': nc,
                        'effective_query_count': '' if macro else n,
                        'effective_class_count': n if macro else '',
                        'aggregation_unit': 'class' if macro else 'query',
                        'count_status': 'reported' if n != '' else 'not_reported',
                        'source': report['source']['path'],
                        'source_sha256': report['source']['sha256'],
                    })
    return rows


def generated_tables(selection, workspace):
    catalog = workspace / 'wbm-region-retrieval/results_catalog'
    if sha(catalog / 'snapshot.json') != selection['source_catalog_sha256']:
        raise ValueError('The frozen catalogue snapshot changed')
    with (catalog / 'metrics.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        wanted = set(selection['b_keys'] + selection['c_keys'])
        rows = [row for row in reader if row['key'] in wanted]
    with (catalog / 'registry.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        registry_fields = reader.fieldnames
        registry = [row for row in reader if row['key'] in wanted]
    found = {row['key'] for row in rows}
    if found != wanted:
        raise ValueError('Missing selected catalogue keys: ' + str(wanted - found))
    output = {'provenance/registry_subset.csv': csv_bytes(registry_fields, registry)}
    for section, keys in [('B', selection['b_keys']), ('C', selection['c_keys'])]:
        subset = [row for row in rows if row['key'] in keys]
        output[f'{section}/tables/all_metrics.csv'] = csv_bytes(fields, subset)
        for protocol, name in [('retrieval_v1', 'ordinary'),
                               ('coverage_cross_combination_v1', 'cross_combination')]:
            selected = [r for r in subset if r['protocol'] == protocol
                        and r['split'] == 'test' and r['group'] == 'all']
            if not selected:
                raise ValueError('Empty paper table: ' + section + '/' + name)
            output[f'{section}/tables/{name}_all_k.csv'] = csv_bytes(fields, selected)
            short = [r for r in selected if (r['metric'], r['K']) in {
                ('Precision', '10'), ('mAP', '100'), ('Coverage-nDCG', '10'),
                ('MeanCoverage', '10'), ('FullCoverageHit', '10'),
                ('FullCoveragePrecision', '10'), ('FullCoverageRecall', '100'),
                ('FullCoverage_mAP', '100')}]
            output[f'{section}/tables/{name}_paper_metrics.csv'] = csv_bytes(fields, short)
    frows = [r for r in rows if r['key'] in selection['b_keys'] and r['group'] != 'all']
    output['F/tables/group_and_class_macro_metrics.csv'] = csv_bytes(fields, frows)
    output['F/tables/group_and_class_macro_paper_metrics.csv'] = csv_bytes(fields, [
        r for r in frows if (r['metric'], r['K']) in {
            ('Precision', '10'), ('mAP', '100'), ('Coverage-nDCG', '10'),
            ('MeanCoverage', '10'), ('FullCoverageHit', '10'),
            ('FullCoveragePrecision', '10'), ('FullCoverageRecall', '100'),
            ('FullCoverage_mAP', '100')}])
    output['F/tables/k_curves_source.csv'] = csv_bytes(fields, [
        r for r in rows if r['key'] in selection['b_keys'] and r['group'] == 'all'
        and r['K'] in {'1', '5', '10', '20', '50', '100'}])
    snapshot = json.loads((catalog / 'snapshot.json').read_text())
    snapshot['runs'] = [r for r in snapshot['runs'] if r['key'] in wanted]
    counts = metric_count_rows(snapshot['runs'])
    identity = lambda row: tuple(row[k] for k in ('key', 'metric_version', 'group', 'metric', 'K'))
    if {identity(r) for r in counts} != {identity(r) for r in rows}:
        raise ValueError('Metric denominators do not match the selected catalogue rows')
    for section, keys in [('B', selection['b_keys']), ('C', selection['c_keys']),
                          ('F', selection['b_keys'])]:
        output[f'{section}/tables/metric_counts.csv'] = csv_bytes(list(counts[0]), [
            r for r in counts if r['key'] in keys])
    snapshot.pop('inventory', None)
    snapshot['export_note'] = 'Selected existing records; not a new official catalogue.'
    output['provenance/catalog_snapshot_subset.json'] = json_bytes(snapshot)
    return output


def build(selection_path, workspace, output):
    selection = json.loads(selection_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'package_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['selection_sha256'] != sha(selection_path):
            raise ValueError('Existing export has a different selection; preserve it before rebuilding')
        return verify(output)
    files, destinations = [], set()
    for entry in selection['files']:
        target = entry['target']
        if target in destinations:
            raise ValueError('Duplicate target: ' + target)
        destinations.add(target)
        source, dest = inside(workspace, entry['source']), inside(output, target)
        if source.is_symlink() or not source.is_file() or source.name.startswith('.'):
            raise ValueError('Invalid publication source: ' + str(source))
        stat = source.stat()
        if sha(source) != entry['sha256'] or stat.st_size != entry['size']:
            raise ValueError('Source changed: ' + str(source))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_symlink() or sha(dest) != entry['sha256']:
                raise ValueError('Refuse to overwrite different target: ' + str(dest))
        else:
            shutil.copy2(source, dest)
        if sha(dest) != entry['sha256']:
            raise ValueError('Copy mismatch: ' + str(dest))
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Source changed while copying: ' + str(source))
        files.append(dict(entry, kind='source_copy'))
    derived = generated_tables(selection, workspace)
    derived['provenance/selection.json'] = selection_path.read_bytes()
    for target, payload in derived.items():
        if target in destinations:
            raise ValueError('Generated target conflicts with source: ' + target)
        write_bytes(inside(output, target), payload)
        files.append({'target': target, 'size': len(payload),
                      'sha256': hashlib.sha256(payload).hexdigest(), 'kind': 'derived',
                      'source': 'wbm-region-retrieval/results_catalog/{metrics.csv,registry.csv,snapshot.json}',
                      'purpose': 'Exact filtering of existing catalogue; proportions retained'})
    display = selection['display']
    write_bytes(output / 'paper_use.json', json_bytes(display))
    files.append({'target': 'paper_use.json', 'size': (output/'paper_use.json').stat().st_size,
                  'sha256': sha(output/'paper_use.json'), 'kind': 'derived',
                  'source': 'provenance/selection.json', 'purpose': 'Publication entry points'})
    manifest = {'schema_version': 1, 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'kind': 'publication_material_export', 'new_experiment': False,
                'new_formal_evaluations': 0, 'efficiency_experiments': 'omitted_by_user',
                'source_workspace': str(workspace), 'selection_sha256': sha(selection_path),
                'scope': 'A/B/C/D core evidence; F existing statistics; G optional auxiliary evidence',
                'marker_policy': 'Copied source_completed.json files retain original-run paths; '
                                 'only this package_manifest seals the exported subset.',
                'excluded': ['training checkpoints', 'full pair-score matrices', 'full rankings',
                             'superseded figure drafts', 'new efficiency measurements'],
                'files': sorted(files, key=lambda row: row['target'])}
    write_bytes(manifest_path, json_bytes(manifest))
    return verify(output)


def verify(output):
    manifest = json.loads((output / 'package_manifest.json').read_text())
    targets = set()
    total = 0
    for row in manifest['files']:
        if row['target'] in targets:
            raise ValueError('Duplicate target')
        targets.add(row['target'])
        path = inside(output, row['target'])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            raise ValueError('Missing/corrupt material: ' + str(path))
        total += row['size']
    display = json.loads((output / 'paper_use.json').read_text())
    for section, entries in display.items():
        for row in entries:
            if row['path'] not in targets:
                raise ValueError('Unsealed paper entry: ' + row['path'])
    return {'status': 'verified', 'files': len(targets), 'bytes': total,
            'manifest_sha256': sha(output/'package_manifest.json'),
            'scope': 'Every selected file exists and matches byte hash; source and derived-copy audit, '
                     'not a new model evaluation or a complete archived-run reconstruction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'verify'])
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'build':
        if args.selection is None or args.workspace is None:
            parser.error('build requires --selection and --workspace')
        result = build(args.selection, args.workspace.resolve(), args.output.resolve())
    else:
        result = verify(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
