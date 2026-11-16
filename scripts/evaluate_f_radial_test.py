#!/usr/bin/env python3
"""One frozen radial beta=1 F test, with exact_v1 and coverage_v1 reports."""
from __future__ import annotations
import argparse
import csv
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'valid-run', 'data-file', 'compared-root', 'implementation', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    source, valid, output = a.source_run.resolve(), a.valid_run.resolve(), a.output_dir.resolve()
    compared = a.compared_root.resolve()
    sys.path.insert(0, str(compared))
    from wbm_retrieval.common import read_json, write_json, file_hash, digest, get_device
    from wbm_retrieval.data import Protocol, source_fingerprint
    from wbm_retrieval.training import load_checkpoint, parameter_hash
    from wbm_retrieval.diagnostics import batched_map_scores
    from wbm_retrieval.matching import greedy_assignment
    from wbm_retrieval.exact_evaluation import evaluate_suite_rankings
    from wbm_retrieval.exact_metrics import order_scores
    from wbm_retrieval.regions import extract_regions
    from experiments.proposed._shared.ef_suite import build_model, embeddings
    from shared.coverage_metrics import evaluate_rankings_file
    if output.exists() or any(output == x or x in output.parents or output in x.parents for x in (source, valid)):
        raise ValueError('Use a fresh output outside existing source/valid runs')
    decision = read_json(valid/'selection_before_combination.json')
    plan = read_json(valid/'plan_before_evaluation.json')
    completed = read_json(valid/'completed.json')
    assert completed['status'] == 'completed' and decision['selected_beta'] == 1
    assert decision['selected_radial_variant'] == 'radial_b1' and plan['radial_bins'] == 16
    assert plan['beta_grid'] == [0, 1, 2, 4] and decision['selection_metric'] == 'mAP@10'
    assert file_hash(a.implementation) == plan['script_sha256']
    for name in ('selection_before_combination.json', 'plan_before_evaluation.json', 'baseline_h512.npy',
                 'radial_histograms.npy', 'radial_b1/rankings.csv'):
        assert file_hash(valid/name) == completed['files'][name], name
    spec = importlib.util.spec_from_file_location('f_frozen_radial_definition', a.implementation)
    radial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(radial)
    manifest = read_json(source/'suite_manifest.json')
    prepared = read_json(source/'cache/prepared.json')
    assert manifest['signature'] == prepared['signature'] == digest({k:v for k,v in manifest.items() if k != 'signature'})
    config = read_json(source/'seed_2026/F/run_manifest.json')['config']
    assert config['matching'] == manifest['settings']['matching']
    assert config['matching']['shape_threshold'] == config['matching']['score_threshold'] == 0
    assert config['matching']['lambda_pos'] == config['matching']['lambda_scale'] == 0
    assert source_fingerprint(config) == manifest['sources']
    for name, sha in manifest['code'].items():
        assert file_hash(ROOT/name) == sha, name
    assert file_hash(a.data_file) == plan['data_sha256'] == manifest['sources']['data_file']
    assert file_hash(ROOT/'wbm_retrieval/exact_metrics.py') == file_hash(compared/'shared/exact_metrics.py')
    cache_names = ('cache/test_regions.npz', 'cache/test_F.npy', 'cache/test_masks.npy', 'cache/labels.npy')
    for name in cache_names:
        assert file_hash(source/name) == prepared['files'][name], name
    checkpoint_path = source/'seed_2026/F/best_nce.pt'
    assert file_hash(checkpoint_path) == plan['checkpoint_sha256']
    ckpt = load_checkpoint(checkpoint_path)
    assert ckpt['signature'] == digest({'suite': manifest['signature'], 'seed': 2026, 'variant':'F'})
    input_spec = manifest['suite']['variants']['F']
    assert ckpt['input_spec'] == input_spec and ckpt['variant'] == 'F'
    assert read_json(source/'seed_2026/F/training_summary.json')['completed']
    protocol = Protocol(config)
    pool = protocol.candidates
    assert len(protocol.queries) == 7199 and pool.candidates_per_query == 7198
    for name in ('test_queries.csv', 'test_gallery.csv', 'metadata.json'):
        path = ROOT/'protocol/retrieval_v1'/name
        assert file_hash(path) == manifest['sources']['retrieval_protocol/'+name]
        assert file_hash(path) == file_hash(compared/'protocol/retrieval_v1'/name)
    with np.load(source/'cache/test_regions.npz') as z:
        meta = dict(z)
    np.testing.assert_array_equal(meta['map_ids'], pool.gallery)
    labels = np.load(source/'cache/labels.npy')
    model = build_model(config, input_spec).to(get_device(a.device))
    model.load_state_dict(ckpt['model'])
    model.requires_grad_(False)
    model_hash = parameter_hash(model)
    protected = [source/'suite_manifest.json', source/'cache/prepared.json', checkpoint_path,
                 source/'seed_2026/F/test/best_nce/metrics.json', source/'seed_2026/F/test/best_nce/rankings.csv']
    protected += [source/n for n in cache_names] + [ROOT/n for n in manifest['code']]
    protected += [valid/n for n in ('selection_before_combination.json','plan_before_evaluation.json','completed.json')]
    snapshot = {str(f): {'sha256':file_hash(f), 'mtime_ns':f.stat().st_mtime_ns} for f in protected}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run = {'method':'F_radial_beta1', 'beta':1, 'bins':16, 'rotation_pooling':False, 'bidirectional':False,
        'checkpoint':str(checkpoint_path), 'checkpoint_sha256':file_hash(checkpoint_path), 'epoch':ckpt['epoch'],
        'valid_selection':str(valid/'selection_before_combination.json'), 'selection_metric':'valid mAP@10',
        'selection_uses_valid_labels':True, 'parameter_search_on_test':False, 'training_required':False,
        'query_count':7199, 'candidates_per_query':7198, 'total_directed_pairs':7199*7198,
        'ranking_limit':500, 'candidate_rule':'all_gallery_except_query',
        'retrieval_protocol_signature':protocol.retrieval.signature, 'config':config,
        'implementation_sha256':file_hash(a.implementation), 'test_script_sha256':file_hash(__file__),
        'coverage_engine_sha256':file_hash(compared/'shared/coverage_metrics.py'),
        'source_snapshot':snapshot, 'model_parameter_sha256':model_hash}
    write_json(output/'selection_before_test.json', run)
    # Replay selected valid ranks before any new test inference.
    with np.load(source/'cache/valid_regions.npz') as z:
        vm = dict(z)
    vf, vh = np.load(valid/'baseline_h512.npy'), np.load(valid/'radial_histograms.npy')
    vp = read_json(source/'valid_pool.json')
    selected = set(int(x) for x in np.array(vp['queries'])[np.linspace(0,len(vp['queries'])-1,32,dtype=int)])
    old = {q:[] for q in selected}
    with (valid/'radial_b1/rankings.csv').open() as f:
        for row in csv.DictReader(f):
            q = int(row['query_id'])
            if q in old: old[q].append(int(row['candidate_id']))
    vi = {int(i):np.arange(s,e) for i,s,e in zip(vm['map_ids'],vm['offsets'][:-1],vm['offsets'][1:])}
    width = int(np.diff(vm['offsets']).max())
    pad = np.full((len(vm['map_ids']),width),-1,dtype=int)
    for row,i in enumerate(vm['map_ids']): pad[row,:len(vi[int(i)])]=vi[int(i)]
    for q in selected:
        qt = vi[q]
        sim = np.clip(vf[qt]@vf.T,0,1)*np.exp(-radial.radial_distance(vh[qt,None],vh[None]))
        scores = batched_map_scores(sim,pad,vm['area'][qt],config['matching'])
        keep = vm['map_ids'] != q
        ids, scores = vm['map_ids'][keep], scores[keep]
        order = order_scores(ids,scores,q,2026)[:500]
        np.testing.assert_array_equal(ids[order],old[q])
    write_json(output/'valid_replay_audit.json',{'queries':len(selected),'top500_identical':True})
    del vf, vh, vm, sim
    features = embeddings(model, source/'cache/test_F.npy', config, get_device(a.device))
    np.save(output/'test_h512.npy',features)
    masks = np.load(source/'cache/test_masks.npy',mmap_mode='r')
    assert len(features) == len(masks) == meta['offsets'][-1]
    indices = {int(i):np.arange(s,e) for i,s,e in zip(meta['map_ids'],meta['offsets'][:-1],meta['offsets'][1:])}
    with np.load(a.data_file,allow_pickle=False) as z:
        maps = z['arr_0'][meta['map_ids']]
        np.testing.assert_array_equal(z['arr_1'][meta['map_ids']], labels[meta['map_ids']])
    hist = np.empty((len(masks),16))
    for raw,i in zip(maps,meta['map_ids']):
        regions = extract_regions(raw,**config['proposal'])
        assert len(regions)==len(indices[int(i)])
        for region,t in zip(regions,indices[int(i)]):
            np.testing.assert_array_equal(region['mask'],masks[t])
            assert int(masks[t].sum())==meta['area'][t]
            hist[t]=radial.radial_histogram(masks[t],np.isin(raw,[1,2]))
    np.save(output/'test_radial_histograms.npy',hist)
    width = int(np.diff(meta['offsets']).max())
    padded=np.full((len(meta['map_ids']),width),-1,dtype=int)
    positions={int(i):r for r,i in enumerate(meta['map_ids'])}
    for i,tokens in indices.items(): padded[positions[i],:len(tokens)]=tokens
    # Independent scalar parity on 1,024 region-set pairs before full retrieval.
    ids=meta['map_ids'][np.linspace(0,len(meta['map_ids'])-1,32,dtype=int)]
    max_error=0.
    for q in ids:
        qt=indices[int(q)]
        sims=np.clip(features[qt]@features.T,0,1)*np.exp(-radial.radial_distance(hist[qt,None],hist[None]))
        actual=batched_map_scores(sims,padded[[positions[int(c)] for c in ids]],meta['area'][qt],config['matching'])
        expected=[]
        for c in ids:
            ct=indices[int(c)];m=np.empty((len(qt),len(ct)))
            for r,ti in enumerate(qt):
                for s,tj in enumerate(ct):
                    d=sum(abs(float(hist[ti,:k].sum())-float(hist[tj,:k].sum())) for k in range(1,17))/16
                    m[r,s]=np.clip(np.dot(features[ti],features[tj]),0,1)*np.exp(-d)
            weights=np.sqrt(meta['area'][qt])
            expected.append(sum(weights[r]*v for r,s,v in greedy_assignment(m))/weights.sum() if len(qt) else 0.)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        max_error=max(max_error,float(np.max(np.abs(actual-expected))))
    write_json(output/'geometry_matching_audit.json', {'test_maps':len(maps),'regions':len(masks),
        'all_masks_reextracted_equal':True,'scalar_batched_pairs':1024,'max_error':max_error})
    target=output/'test';target.mkdir()
    with (target/'rankings.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['query_id','rank','candidate_id','similarity_score'])
        def rankings():
            for number,q in enumerate(protocol.queries,1):
                c=np.asarray(pool[q]);qt=indices[int(q)]
                sim=np.clip(features[qt]@features.T,0,1)*np.exp(-radial.radial_distance(hist[qt,None],hist[None]))
                scores=batched_map_scores(sim,padded[[positions[int(i)] for i in c]],meta['area'][qt],config['matching'])
                assert np.isfinite(scores).all()
                order=order_scores(c,scores,q,config['evaluation']['tie_seed'])[:500]
                writer.writerows((int(q),rank,int(c[i]),float(scores[i])) for rank,i in enumerate(order,1))
                yield int(q),c[order]
                if number%250==0:print(f'Test retrieval {number}/7199 ({time.monotonic()-started:.1f}s)',flush=True)
        exact=evaluate_suite_rankings(config,labels,pool,rankings(),target,'test',split_ids=meta['map_ids'],
            metadata={'method':'F_radial_beta1','beta':1,'representation':'h512','aggregation':'original_F',
            'checkpoint_sha256':file_hash(checkpoint_path),'retrieval_protocol_signature':protocol.retrieval.signature,
            'candidates_scored':'full_pool','ranking_limit':500,'bootstrap_samples':0})
    taxonomy=sorted({tuple(labels[i]) for i in protocol.ids['train']})
    mapping={x:i for i,x in enumerate(taxonomy)}
    class_ids=np.full(len(labels),-1,dtype=int)
    for q in meta['map_ids']:class_ids[q]=mapping[tuple(labels[q])]
    print('Exact complete; computing coverage on identical rankings',flush=True)
    coverage=evaluate_rankings_file(SimpleNamespace(labels=labels,class_ids=class_ids),pool,target/'rankings.csv',
        config['evaluation']['ks'],target,split_name='test',metadata={
        'method':'F_radial_beta1','beta':1,'checkpoint_sha256':file_hash(checkpoint_path),
        'ranking_source_sha256':file_hash(target/'rankings.csv'),'candidate_rule':'all_gallery_except_query',
        'ranking_limit':500,'smoke':False,'retrieval_protocol_signature':protocol.retrieval.signature})
    assert parameter_hash(model)==model_hash
    for name,state in snapshot.items():
        f=Path(name);assert file_hash(f)==state['sha256'] and f.stat().st_mtime_ns==state['mtime_ns'],name
    write_json(output/'completed.json',{'status':'completed','query_count':7199,'candidates_per_query':7198,
        'beta':1,'source_files_unchanged':True,'model_unchanged':True,'seconds':time.monotonic()-started,
        'files':{str(f.relative_to(output)):file_hash(f) for f in output.rglob('*') if f.is_file()}})
    print({'exact':exact['metrics'],'coverage':coverage['metrics']},flush=True)


if __name__=='__main__': main()
