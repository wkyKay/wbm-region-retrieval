"""Read-only model, original-mask geometry and canonical historical reports."""
from pathlib import Path
from f_region_evidence.inputs import bind_sealed
from f_region_score_search.inputs import _bind
from f_region_wbbs.protocol import BASE_PROTOCOL_SIGNATURE, FROZEN, RULE, PROTOCOL
from paper_c_ablation import inputs as historical
from cnn_augmentation.evaluation import token_layout
from .state import ROOT, SOURCE, GRID, THETA0, KS, read_json, read_signed, sha, digest, check_binding


def reference_row(folder, model, stage, variant, *, canonical=None):
    folder = Path(folder).resolve()
    binding = {}
    bind_sealed(folder, binding)
    meta = read_json(folder/'evaluation_manifest.json')
    check = read_json(folder/'verification.json')
    identity = read_json(folder/'run_identity.json')
    split = stage.removeprefix('cross_')
    cross = stage.startswith('cross_')
    nq, ng = (500, 3599) if split == 'valid' else (7199, 7199)
    pairs = FROZEN[split]['candidate_pairs'] if cross else nq*(ng-1)
    protocol = FROZEN[split]['derived_signature'] if cross else BASE_PROTOCOL_SIGNATURE
    if (meta['signature'] != read_json(folder/'completed.json')['signature']
            or identity['evaluation_signature'] != meta['signature']
            or meta['key'] != identity['key']
            or meta.get('split') != split or meta.get('ks') != KS or meta.get('tie_seed') != 2026
            or meta.get('num_queries') != nq or meta.get('gallery_count') != ng
            or meta.get('candidate_rule') != (RULE if cross else 'all_gallery_except_query')
            or meta.get('retrieval_protocol_signature') != protocol
            or check.get('status') != 'verified' or check.get('num_queries') != nq
            or check.get('full_candidates_checked') != pairs or check.get('ranking_rows') != nq*500
            or check.get('all_queries_full_candidate_topk_exact') is not True
            or check.get('ranking_sha256') != sha(folder/'rankings.csv')):
        raise ValueError('Historical report lacks the exact full-candidate protocol proof: '+str(folder))
    if canonical is not None:
        expected = canonical.get('completed_sha256', canonical.get('source_completed_sha256'))
        if canonical['key'] != identity['key'] or canonical['signature'] != meta['signature'] or expected != sha(folder/'completed.json'):
            raise ValueError('Canonical reference identity changed')
    elif meta.get('spec') != GRID[variant] or meta.get('variant') != variant:
        raise ValueError('Historical weight reference differs from requested parameters')
    row = {'model':model, 'stage':stage, 'split':split, 'variant':variant, 'spec':GRID[variant],
        'path':str(folder), 'key':identity['key'], 'run_id':identity['run_id'], 'signature':meta['signature'],
        'register':False, 'historical_reference':True, 'completed_sha256':sha(folder/'completed.json'),
        'source_binding':binding, 'source_spec':meta.get('spec'),
        'coverage_metrics':str(folder/'coverage_metrics.json'), 'metric_query_counts':{}}
    for family,name in ([('coverage','coverage_metrics.json')] if cross else [('exact','metrics.json'),('coverage','coverage_metrics.json')]):
        report = read_json(folder/name)
        if any(report.get('metadata',{}).get(k) != v for k,v in meta.items()):
            raise ValueError('Historical metrics detached from their original manifest')
        row['metric_query_counts'][family] = report['metric_query_counts']
    if not cross:
        row['metrics'] = str(folder/'metrics.json')
    return row


def load(split='valid'):
    if split not in ('valid','test'):
        raise ValueError('Unknown frozen split')
    data = historical.load(split)
    binding = dict(data['source_binding'])
    source_manifest = read_signed(SOURCE/'run_manifest.json')
    root_marker = read_json(SOURCE/'completed.json')
    if root_marker.get('status') != 'completed' or root_marker['signature'] != source_manifest['signature']:
        raise ValueError('Rotation source experiment is not sealed')
    _bind(SOURCE/'completed.json', binding)
    def anchored(path):
        path = Path(path).resolve()
        _bind(path, binding, root_marker['files'][str(path.relative_to(SOURCE))])
    anchored(SOURCE/'run_manifest.json')
    anchored(SOURCE/'frozen_plan.json')
    feature_folder = SOURCE/'features/rotation'/split
    anchored(feature_folder/'completed.json')
    feature = read_signed(feature_folder/'manifest.json')
    bind_sealed(feature_folder,binding,feature['feature_signature'])
    checkpoint = SOURCE/'training/rotation/checkpoint.pt'
    for path in [checkpoint, SOURCE/'training/rotation/training_summary.json', SOURCE/'training/rotation/run_manifest.json', SOURCE/'training/rotation/completed.json']:
        anchored(path)
    summary = read_json(SOURCE/'training/rotation/training_summary.json')
    if (feature['model'] != 'rotation' or feature['split'] != split or feature['views'] != ['original']
            or feature['parent_context'] != source_manifest['signature']
            or feature['checkpoint_path'] != str(checkpoint) or feature['checkpoint_sha256'] != sha(checkpoint)
            or feature['checkpoint']['epoch'] != 100 or summary['epochs_completed'] != 100
            or summary['optimizer_steps'] != 39000 or summary['checkpoint_selection'] != 'fixed_epoch_100'
            or feature['checkpoint']['parameter_sha256'] != summary['final_parameter_sha256']
            or feature['token_layout'] != token_layout(data['cache'])):
        raise ValueError('Rotation checkpoint, feature provenance or region order changed')
    _bind(Path(feature['patch_path']),binding,feature['patch_sha256'])
    features = {'baseline':Path(data['cache']['folder'])/'h512.npy', 'rotation':feature_folder/'original.npy'}
    for p in features.values(): _bind(p,binding)
    references = {}
    for model in ('baseline','rotation'):
        source_index_path = SOURCE/'evaluation'/model/f'evaluation_index_{split}.json'
        anchored(source_index_path)
        index = read_json(source_index_path)
        if index['context_signature'] != source_manifest['signature'] or index['model'] != model or index['split'] != split:
            raise ValueError('Historical theta0 index context differs')
        for stage in (split,'cross_'+split):
            item = index['evaluations'][stage+'/fusion']
            row = reference_row(item['path'],model,stage,THETA0,canonical=item)
            references[model+'/'+stage+'/'+THETA0] = row
            binding.update(row['source_binding'])
    root = historical.VALID_SOURCE if split=='valid' else historical.TEST_SOURCE
    for name in GRID:
        key = 'baseline/'+split+'/'+name
        if key in references: continue
        folder = root/split/name
        if (folder/'completed.json').is_file():
            row = reference_row(folder,'baseline',split,name)
            references[key] = row
            binding.update(row['source_binding'])
        elif split == 'valid':
            raise ValueError('All twelve historical ordinary-valid controls must exist')
    data.update(source_binding=dict(sorted(binding.items())), features=features, references=references,
                model_source={'rotation':{'feature_manifest':feature,'training_summary':summary},
                              'baseline':{'feature_sha256':sha(features['baseline']), 'source':'unchanged original cache'}})
    return data


def persist_inputs(output, data, split, context_signature):
    from .state import signed, record
    value = signed({'context_signature':context_signature, 'split':split,
        'source_binding':data['source_binding'], 'model_source':data['model_source'],
        'features':{k:str(v.resolve()) for k,v in data['features'].items()},
        'token_layout':token_layout(data['cache']), 'references':data['references']})
    record(Path(output)/('inputs_'+split+'.json'),value)
    return value
