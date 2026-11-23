"""Aligned epoch-100 shape/context diagnostics; never used for model selection.

Wafer labels are read only after the encoder and negative graph are frozen.
The legal context interventions preserve target <= defects <= support and are
input-sensitivity probes, not assumed label-preserving semantic counterfactuals.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from f_region_structure.state import record, seal, verified
from f_representation_audit.context import build_context_views
from f_representation_audit.geometry import audit_geometry, paired_bootstrap
from f_representation_audit.run import (context_selection, csv, evaluate_context, inference,
                                       original_windows, select_singletons, state_hash)
from wbm_retrieval.common import digest, file_hash, read_json

MODELS = ('baseline', 'context_dropout', 'filtered_negatives')
KS = (1,5,10,20,50,100,500)


def dispersion(features):
    """Descriptive collapse diagnostics on the same aligned region cohort."""
    values=np.asarray(features,dtype=np.float64)
    if values.ndim!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Dispersion requires finite feature vectors')
    norms=np.linalg.norm(values,axis=1)
    if np.any(norms<=0):
        raise ValueError('Zero-norm representation detected')
    normalized=values/norms[:,None]
    centered=normalized-normalized.mean(0)
    singular=np.linalg.svd(centered,full_matrices=False,compute_uv=False)
    power=singular**2
    total=float(power.sum())
    probabilities=power[power>0]/total if total>0 else np.array([])
    effective=float(np.exp(-np.sum(probabilities*np.log(probabilities)))) if len(probabilities) else 0.
    return {'regions':len(values),'width':values.shape[1],
        'raw_mean_dimension_variance':float(np.var(values,axis=0).mean()),
        'normalized_mean_dimension_variance':float(np.var(normalized,axis=0).mean()),
        'normalized_covariance_trace':total/len(values),'covariance_effective_rank':effective,
        'numerical_rank':int(np.linalg.matrix_rank(centered)),
        'zero_normalized_variance':bool(total<=1e-12),
        'definition':'population covariance of L2-normalized region vectors; effective rank exp(entropy(normalized covariance eigenvalues)); descriptive, no tuned threshold'}


def context_effect(reference, candidate, *, repetitions=2000, seed=2026):
    """Paired map-cluster bootstrap, stratified by complete wafer combination.

    Retain the identical non-noop context cohort for every encoder. Rotation
    distances stay in the paired table, including rotation no-ops.
    """
    if type(repetitions) is not int or repetitions<1:
        raise ValueError('Bootstrap repetitions must be positive')
    fields=('cosine_distance','rotation_distance','distance_minus_rotation','larger_than_rotation')
    def index(rows):
        result={}
        for row in rows:
            key=(row['space'],row['view'],int(row['map_id']),int(row['region_id']))
            if key in result:
                raise ValueError('Duplicate context probe identity')
            result[key]=row
        return result
    left,right=index(reference),index(candidate)
    if left.keys()!=right.keys():
        raise ValueError('Context comparisons require identical regions and views')
    grouped=defaultdict(lambda:defaultdict(list))
    common={(a['space'],a['map_id'],a['region_id']) for a in left.values()
            if a['view']=='context_drop' and not a['is_noop'] and a['has_context']}
    for key,a in left.items():
        b=right[key]
        for name in ('token_index','label_count','label_signature','is_noop','has_context',
                     'valid_region_input','source_context_pixels','source_target_pixels',
                     'source_all_defect_changed_pixels'):
            if a[name]!=b[name]:
                raise ValueError('Context model comparison changed its input cohort')
        if not a['valid_region_input']:
            raise ValueError('Only legal context inputs are allowed')
        if a['view']=='rot90' or (a['space'],a['map_id'],a['region_id']) not in common:
            continue
        for group in ('all','single' if a['label_count']==1 else 'mixed'):
            grouped[(a['space'],a['view'],group)][(a['label_signature'],a['map_id'])].append(
                [float(b[f])-float(a[f]) for f in fields])
    results={}
    for key,maps in sorted(grouped.items()):
        strata=defaultdict(list)
        for (label,_),rows in sorted(maps.items()):
            strata[label].append(np.mean(rows,axis=0))
        rng=np.random.default_rng(seed)
        draws=np.zeros((repetitions,len(fields)))
        estimates=[]
        for values in strata.values():
            values=np.asarray(values)
            estimates.append(values.mean(0))
            for start in range(0,repetitions,128):
                count=min(128,repetitions-start)
                sampled=rng.integers(len(values),size=(count,len(values)))
                draws[start:start+count]+=values[sampled].mean(1)/len(strata)
        estimate=np.mean(estimates,axis=0)
        low,high=np.quantile(draws,[.025,.975],axis=0)
        results['/'.join(key)]={'maps':len(maps),'classes':len(strata),
            'regions':sum(map(len,maps.values())), 'effects':{f:{'difference':float(estimate[i]),
                'ci_low':float(low[i]),'ci_high':float(high[i]),'improvement_direction':'lower'}
                for i,f in enumerate(fields)}}
    return {'difference':'candidate minus baseline', 'repetitions':repetitions, 'seed':seed,
            'resampling':'paired map clusters within complete wafer-label combination; equal combination means',
            'cohort':'same regions with nonzero actual context dropout for all interventions and models',
            'scope':'conditional on frozen models, fixed selected maps and fixed legal interventions; no training-seed uncertainty',
            'results':results}


def _load_model(inputs, path, device):
    import torch
    from experiments.proposed._shared.ef_suite import build_model
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('epoch')!=100 or 'model' not in checkpoint:
        raise ValueError('Diagnostics use prespecified epoch100 checkpoints only')
    model=build_model(inputs['config'],inputs['spec']).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    return model


def run_diagnostics(output, inputs, checkpoint_paths, context_signature, *, device='cuda:0',
                    batch_size=256, bootstrap=2000, train_per_class=128,
                    context_per_combination=4, seed=2026):
    """Run three frozen encoders on identical train/valid diagnostic samples.

    ``inputs`` is f_representation_audit.inputs.load_inputs(...) output. Paths
    map baseline/context_dropout/filtered_negatives to their epoch100 checkpoint.
    No test input is opened and no checkpoint is selected by this function.
    """
    if set(checkpoint_paths)!=set(MODELS) or not context_signature:
        raise ValueError('All three frozen checkpoint identities are required')
    if (type(bootstrap) is not int or bootstrap<1 or type(batch_size) is not int or batch_size<1
            or seed!=2026 or train_per_class!=128 or context_per_combination!=4):
        raise ValueError('Invalid diagnostic recipe or changed prespecified cohorts')
    output=Path(output)/'diagnostics'
    datasets,labels=inputs['datasets'],inputs['labels']
    selections={
        'train_representative':('train',select_singletons(datasets['train'],train_per_class,seed)),
        'valid_representative':('valid',select_singletons(datasets['valid'],None,seed)),
        'valid_all_regions':('valid',np.flatnonzero(datasets['valid']['singleton_labels']>=0))}
    indices=context_selection(datasets['valid'],labels,context_per_combination,seed)
    project=Path(__file__).resolve().parents[1]
    dependencies=[Path(__file__).resolve(),project/'f_representation_audit/run.py',
                  project/'f_representation_audit/context.py',project/'f_representation_audit/geometry.py',
                  project/'experiments/proposed/_shared/ef_suite.py']
    code={str(path):file_hash(path) for path in dependencies}
    science={'context_signature':context_signature,'version':1,'seed':seed,'code':code,
        'checkpoints':{name:{'path':str(Path(path).resolve()),'sha256':file_hash(path)}
                       for name,path in checkpoint_paths.items()},
        'selections':{name:{'split':split,'indices':selected.tolist()} for name,(split,selected) in selections.items()},
        'context_indices':indices.tolist(),'bootstrap':bootstrap,'ks':list(KS),
        'source_protection_sha256':digest(inputs['protection']),
        'source_files':inputs['protection'],
        'context_views':['original','rot90','target_only','context_drop'],
        'scope':'train and valid diagnostics only; single-label wafer classes are diagnostic proxies, not region-level truth; no checkpoint selection'}
    signature=digest(science)
    if verified(output,signature):
        return read_json(output/'summary.json')
    output.mkdir(parents=True,exist_ok=True)
    record(output/'manifest.json',{**science,'signature':signature})
    windows=original_windows(inputs['maps'],datasets['valid'],indices)
    ds=datasets['valid']
    keys=[f'{int(ds["token_map_ids"][i])}:{int(ds["token_region_ids"][i])}' for i in indices]
    views,audits=build_context_views(windows,keys,seed=seed,include_target_removed_control=False)
    if not np.array_equal(views['original'],np.asarray(ds['patches'][indices],dtype=np.float32)):
        raise ValueError('Raw context windows do not replay original input pixels')
    from experiments.proposed._shared.ef_suite import fixed_window_patch
    positions={int(sid):i for i,sid in enumerate(ds['map_ids'])}
    for i,index in enumerate(indices):
        sid=int(ds['token_map_ids'][index]);rid=int(ds['token_region_ids'][index])
        canonical,_=fixed_window_patch(inputs['maps'][sid],ds['region_masks'][positions[sid]]==rid)
        if not np.array_equal(canonical,views['original'][i]):
            raise ValueError('Canonical original input reconstruction differs')
    csv(output/'context_input_audit.csv',audits)
    np.savez_compressed(output/'context_examples.npz',indices=indices[:12],
                        **{name:values[:12] for name,values in views.items()})
    sample_rows=[]
    for name,(split,selected) in selections.items():
        dataset=datasets[split]
        for index in selected:
            sample_rows.append({'subset':name,'split':split,'token_index':int(index),
                'map_id':int(dataset['token_map_ids'][index]),'region_id':int(dataset['token_region_ids'][index]),
                'singleton_label':int(dataset['singleton_labels'][index])})
    csv(output/'selected_regions.csv',sample_rows)
    geometry,context_rows,context_summary,states={},{},{},{}
    for name in MODELS:
        folder=output/name
        model_signature=digest({'parent':signature,'model':name,'checkpoint':science['checkpoints'][name]})
        if not verified(folder,model_signature):
            folder.mkdir(parents=True,exist_ok=True)
            model=_load_model(inputs,checkpoint_paths[name],device)
            before=state_hash(model)
            for subset,(split,selected) in selections.items():
                dataset=datasets[split]
                features=inference(model,dataset['patches'],selected,device,batch_size)
                np.savez_compressed(folder/(subset+'_features.npz'),indices=selected,**features)
                for space,values in features.items():
                    result=audit_geometry(values,dataset['singleton_labels'][selected],
                                          dataset['token_map_ids'][selected],ks=KS)
                    result['summary']['dispersion']=dispersion(values)
                    record(folder/f'{subset}_{space}.json',result)
                    csv(folder/f'{subset}_{space}_per_region.csv',result['rows'])
            summary,rows=evaluate_context(model,views,audits,indices,ds,labels,device,batch_size,name)
            record(folder/'context_summary.json',summary)
            record(folder/'context_rows.json',rows)
            csv(folder/'context_per_region.csv',rows)
            if state_hash(model)!=before:
                raise ValueError('Diagnostic modified frozen model tensors')
            record(folder/'model_state.json',{'model':name,'epoch':100,'all_state_sha256':before,
                                             'all_model_tensors_unchanged':True})
            seal(folder,model_signature)
            del model
        states[name]=read_json(folder/'model_state.json')
        for subset in selections:
            for space in ('h512','z128'):
                geometry[(name,subset,space)]=read_json(folder/f'{subset}_{space}.json')
        context_rows[name]=read_json(folder/'context_rows.json')
        context_summary[name]=read_json(folder/'context_summary.json')
    comparisons={}
    for name in MODELS[1:]:
        comparisons[name]={'geometry':{},'context':context_effect(context_rows['baseline'],context_rows[name],
                                                                repetitions=bootstrap,seed=seed)}
        for subset in selections:
            for space in ('h512','z128'):
                comparisons[name]['geometry'][subset+'/'+space]=paired_bootstrap(
                    geometry[('baseline',subset,space)],geometry[(name,subset,space)],
                    repetitions=bootstrap,seed=seed)
    record(output/'paired_comparisons.json',comparisons)
    if any(file_hash(info['path'])!=info['sha256'] for info in science['checkpoints'].values()):
        raise ValueError('Checkpoint changed during frozen diagnostic')
    if any(file_hash(path)!=expected for path,expected in code.items()):
        raise ValueError('Diagnostic scientific code changed during run')
    summary={'signature':signature,'context_signature':context_signature,'states':states,
        'geometry':{'/'.join(key):value['summary'] for key,value in geometry.items()},
        'context':context_summary,'comparisons':comparisons,
        'verification':{'source_windows_byte_equal':len(indices),'canonical_replay_equal':len(indices),
                        'only_legal_views':True,'all_models_unchanged':True},
        'interpretation':[
            'Context change sensitivity is not itself retrieval accuracy; assess unchanged ordinary/cross galleries jointly.',
            'Context can contain same-defect fragments; lower sensitivity is not automatically semantic improvement.',
            'Single-label wafer labels are diagnostic proxies, not target-region ground truth.',
            'Paired bootstrap is conditional on this seed and gallery and does not establish cross-seed robustness.']}
    record(output/'summary.json',summary)
    seal(output,signature)
    return summary
