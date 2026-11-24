"""Read-only canonical comparisons for the rotation encoder weight search.

Only valid determines weights. Historical label encodings are verified and
normalized in memory before paired query bootstrap; original CSV bytes stay put.
"""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import re

import numpy as np

from cnn_augmentation.reporting import paired_query_bootstrap, SCOPE
from f_region_structure.state import read_signed, signed, record, seal, verified
from wbm_retrieval.common import digest, file_hash, read_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
THETA0 = 'add_a0p4_b0p25'
GRID = {f'add_a{str(a).replace(".","p")}_b{str(b).replace(".","p")}':
        {'family':'add','alpha':a,'beta':b}
        for a in (.4,.5,.6) for b in (.05,.1,.15,.25)}
REPORT_FILES = {'manifest.json','summary.json','valid_selection.csv','comparison_test.csv',
    'comparison_cross_test.csv','comparison_wbbs.csv','paired_test_comparisons.csv','all_metrics.csv',
    'role_aliases.json','verification.json','completed.json'}
# Canonical completed sources, not their later replay copies. Historical path
# strings inside manifests are preserved, including the former wbm-matching name.
WBBS_SOURCES = {
 'valid/exact': ('compared_methods/artifacts/formal_exact_20261109_120738/evaluation/wbbs_valid','valid','wbbs_valid',
  'eb58ded227cf56b254bb8f41b19a89316636fb45a8dc19ebe7181fd5bb56fdca','0de8e83a7a9f538f04eed84613aa3c7c01f256e811ae8d0b6844aaf2d89f93da'),
 'test/exact': ('compared_methods/artifacts/formal_exact_20261109_120738/evaluation/wbbs_test_r2','test','wbbs_test',
  '83c8b2e951b957d4dde6a9c32eda2b239e4765ce0804bf8b5a228d7affd5249c','d89c34eb2b0c1cecdb9d0c5690d2743263672c303b7fe3436e3d1c61ef6d562c'),
 'test/coverage': ('compared_methods/artifacts/coverage_v1_test_20261109_191909/wbbs','test','wbbs_test',
  '87911188633c1df8fcc4ce3080d57e3a87b8ddc79572cfac073de8f94ea028af','82ab3a546553750d9c6bb978ffd297c830ba1da07ace652c34e17dbd75aa7f63'),
 'cross_test/coverage': ('compared_methods/artifacts/cross_coverage_seven_test_20261111_v1/wbbs','test','cross_wbbs',
  '9146995196c4e28eb6c8162d9b44748ca2199ab76667243828664c76e4a40577','a11fd3b203c21b4c6dd5da83dc7b7b7dad114c5564a394447391691472050a27'),
 'cross_valid/coverage': ('wbm-region-retrieval/artificial/f_region_wbbs_20261121T143545+0800/cross_valid/global_wbbs','',
  'f_region_wbbs_f_region_wbbs_20261121T143545_0800_cross_valid_global_wbbs',
  'ba369d53c897445d6158100308f2d8e2a398192cdffeb6c3040d08d09403c4e0','df917c6febf2554865a38f39717cbd9e9893ee07d72dcd5edbea88de2c4f99b1'),
}
METRICS = {'test':(('exact','Precision@10'),('exact','mAP@100'),('coverage','Coverage-nDCG@10')),
 'cross_test':(('coverage','Coverage-nDCG@10'),('coverage','FullCoveragePrecision@10'),('coverage','FullCoverage_mAP@100'))}


def _bind(path, binding, expected=None):
    path=Path(path).resolve()
    value=file_hash(path)
    if expected is not None and value!=expected:raise ValueError('Source seal/hash mismatch: '+str(path))
    if str(path) in binding and binding[str(path)]!=value:raise ValueError('Source changed during report')
    binding[str(path)]=value
    return path


def _check_binding(binding):
    for path,expected in binding.items():
        if file_hash(Path(path))!=expected:raise ValueError('Bound report source changed: '+path)


def _rows(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len({int(r['query_id']) for r in rows})!=len(rows):raise ValueError('Duplicate query IDs')
    return rows


def label_bits(value):
    text=str(value)
    if re.fullmatch('[01]{8}',text):
        if '1' not in text:raise ValueError('Empty defect label')
        return text
    if not re.fullmatch(r'[0-7](?:\|[0-7])*',text):raise ValueError('Unrecognized label signature')
    indices=[int(x) for x in text.split('|')]
    if indices!=sorted(set(indices)):raise ValueError('Noncanonical label-index signature')
    return ''.join('1' if i in indices else '0' for i in range(8))


def normalize_queries(rows):
    """Validate bijective source class encoding before assigning common classes."""
    classes={};labels={};ids=set();normalized=[];encoding=Counter()
    for source in rows:
        q=int(source['query_id']);cls=int(source['class_id']);bits=label_bits(source['label_signature'])
        if q in ids:raise ValueError('Duplicate original query ID')
        ids.add(q)
        if int(source['label_count'])!=bits.count('1'):raise ValueError('Label count changed')
        if cls in classes and classes[cls]!=bits:raise ValueError('Source class maps to multiple signatures')
        if bits in labels and labels[bits]!=cls:raise ValueError('Source signature maps to multiple classes')
        classes[cls]=bits;labels[bits]=cls
        if int(source['pool_size'])<0:raise ValueError('Invalid pool size')
        encoding['binary_8' if re.fullmatch('[01]{8}',str(source['label_signature'])) else 'pipe_indices']+=1
        normalized.append(dict(source,label_signature=bits))
    mapping={label:i+1 for i,label in enumerate(sorted(labels))}
    for row in normalized:row['class_id']=mapping[row['label_signature']]
    return normalized,{'rows':len(rows),'source_class_to_binary_label':{str(k):v for k,v in sorted(classes.items())},
        'source_encodings':dict(encoding),'canonical_class_assignment':'sorted complete binary signatures, one-based; no constant class-ID offset',
        'canonical_classes':mapping}


def checked_pair(reference,candidate,family):
    """Check full population, labels, pool and relevance denominator before CI."""
    left,ld=normalize_queries(reference);right,rd=normalize_queries(candidate)
    a={int(r['query_id']):r for r in left};b={int(r['query_id']):r for r in right}
    if not a or a.keys()!=b.keys():raise ValueError('Full query identities differ')
    denominator='relevant_in_pool' if family=='exact' else 'full_coverage_in_pool'
    for q in sorted(a):
        x,y=a[q],b[q]
        if x['label_signature']!=y['label_signature']:raise ValueError('Query labels disagree after verified conversion')
        for key in ('label_count','pool_size',denominator):
            if int(x[key])!=int(y[key]):raise ValueError('Query pool or relevance denominator differs: '+key)
    return left,right,{'status':'verified','family':family,'query_count':len(a),
        'query_ids_sha256':digest(sorted(a)),'denominator':denominator,
        'reference_encoding':ld,'candidate_encoding':rd,'all_query_labels_pools_and_denominators_equal':True}


def per_query_metric(metric):
    return metric.replace('FullCoverage_mAP@','FullCoverageAP@').replace('mAP@','AP@')


def _family(root,subdir,family,binding,marker_hash=None,manifest_hash=None):
    root=Path(root);folder=root/subdir
    marker=read_json(_bind(root/'completed.json',binding,marker_hash))
    manifest=read_json(_bind(root/'evaluation_manifest.json',binding,manifest_hash))
    if marker.get('status') not in (None,'completed') or not isinstance(marker.get('files'),dict):
        raise ValueError('Canonical evaluation is not completed')
    if marker.get('evaluation_signature',marker.get('signature'))!=manifest.get('signature',marker.get('evaluation_signature')):
        raise ValueError('Canonical manifest signature differs from completion')
    prefix='' if family=='exact' else 'coverage_'
    files={}
    for key,name in [('metrics',prefix+'metrics.json'),('per_query',prefix+'per_query.csv'),('per_class',prefix+'per_class.csv')]:
        path=folder/name;relative=path.relative_to(root).as_posix()
        if relative not in marker['files']:raise ValueError('Report not included in source seal: '+relative)
        files[key]=str(_bind(path,binding,marker['files'][relative]))
    return {**files,'marker':str(root/'completed.json'),'manifest':str(root/'evaluation_manifest.json'),
        'marker_sha256':file_hash(root/'completed.json'),'manifest_sha256':file_hash(root/'evaluation_manifest.json')}


def bind_wbbs_references(output):
    output=Path(output);context=read_signed(output/'run_manifest.json')['signature']
    binding={};families={}
    for role,(path,subdir,key,msha,jsha) in WBBS_SOURCES.items():
        source=_family(ROOT.parent/path,subdir,role.split('/')[1],binding,msha,jsha)
        families[role]={**source,'canonical_key':key,'historical_reference':True}
    exact=read_json(families['test/exact']['marker'])
    em=read_json(families['test/exact']['manifest'])
    cm=read_json(families['test/coverage']['manifest'])
    ranking=exact['files']['test/rankings.csv']
    if (cm.get('ranking_source_sha256')!=ranking or cm.get('ranking_manifest_sha256')!=families['test/exact']['manifest_sha256']
            or cm.get('ranking_evaluation_signature')!=em['signature']):
        raise ValueError('WBBS ordinary exact and coverage are not the same sealed ranking')
    result=signed({'context_signature':context,'families':families,'source_binding':dict(sorted(binding.items())),
        'same_ordinary_ranking':{'ranking_sha256':ranking,'exact_manifest_sha256':families['test/exact']['manifest_sha256'],
          'original_ranking_source_string':cm['ranking_source'],'status':'verified'},
        'selection_role':'WBBS is a fixed benchmark reference; it never selects the regional weights'})
    record(output/'wbbs_references.json',result)
    return result


def _loaded_family(paths,stage,family,binding):
    document=read_json(_bind(paths['metrics'],binding));queries=_rows(_bind(paths['per_query'],binding))
    _bind(paths['per_class'],binding)
    expected=500 if stage.endswith('valid') else 7199
    if document['num_queries']!=expected or len(queries)!=expected:raise ValueError('Wrong complete query population')
    if document['split']!=stage.removeprefix('cross_') or document['ks']!=[1,5,10,20,50,100,500]:
        raise ValueError('Report split or K values differ')
    if not stage.startswith('cross_') and any(int(q['pool_size'])!=(3598 if stage=='valid' else 7198) for q in queries):
        raise ValueError('Ordinary report must retain the complete nonself gallery')
    if document['metric_version']!=('exact_v1' if family=='exact' else 'coverage_v1'):
        raise ValueError('Metric family/version changed')
    normalized,encoding=normalize_queries(queries)
    if len(encoding['canonical_classes'])!=37:raise ValueError('Missing full label combination')
    for metric,value in document['metrics'].items():
        column=per_query_metric(metric)
        vals=[float(q[column]) for q in queries if q[column] not in ('',None) and np.isfinite(float(q[column]))]
        if len(vals)!=document['metric_query_counts'][metric]:raise ValueError('Effective query count differs')
        mean=float(np.mean(vals)) if vals else None
        if (value is None)!=(mean is None) or (value is not None and abs(float(value)-mean)>2e-10):
            raise ValueError('Metric mean detached from source queries: '+metric)
    return {'document':document,'queries':queries,'paths':paths,'encoding':encoding}


def _load_row(row,binding):
    stage=row['stage'];folder=Path(row['path'])
    source={};families=('coverage',) if stage.startswith('cross_') else ('exact','coverage')
    for family in families:
        paths=_family(folder,'',family,binding,row['completed_sha256'])
        declared=row['metrics' if family=='exact' else 'coverage_metrics']
        if isinstance(declared,str) and Path(declared).resolve()!=Path(paths['metrics']).resolve():
            raise ValueError('Index points to a different metric source')
        source[family]=_loaded_family(paths,stage,family,binding)
    if read_json(folder/'completed.json').get('signature')!=row['signature']:
        raise ValueError('Index signature detached from source evaluation')
    return {'row':row,'families':source}


def _flat(document,base):
    groups={'overall':document,'class_macro':{'metrics':document['class_macro_metrics'],
      'metric_query_counts':document['metric_query_counts'],'metric_class_counts':document['class_macro_metric_class_counts']},
      'single_label':document['groups']['single_label'],'mixed_label':document['groups']['mixed_label']}
    groups.update({'label_count_'+k:v for k,v in document['groups']['by_label_count'].items()})
    return [{**base,'group':g,'metric':k,'value':v,'effective_n':d.get('metric_query_counts',{}).get(k),
             'effective_classes':d.get('metric_class_counts',{}).get(k)}
            for g,d in groups.items() for k,v in d['metrics'].items()]


def endpoint_roles(theta0,winner):
    """Five display roles; callers deduplicate by actual canonical key/path."""
    return [('old_theta0','baseline',theta0),('new_theta0','rotation',theta0),
            ('old_theta_star','baseline',winner),('new_theta_star','rotation',winner)]


def merge_protocol_tables(tables):
    """One user-facing WBBS table; keep protocols and effective N explicit."""
    ordinary={row['roles']:row for row in tables['test']}
    cross={row['roles']:row for row in tables['cross_test']}
    if (len(ordinary)!=len(tables['test']) or len(cross)!=len(tables['cross_test'])
            or ordinary.keys()!=cross.keys() or 'WBBS' not in ordinary):
        raise ValueError('Ordinary/cross table roles differ or duplicate')
    columns={
        'test':(('Precision@10','ordinary_exact_P10'),('mAP@100','ordinary_exact_mAP100'),
                ('Coverage-nDCG@10','ordinary_coverage_nDCG10')),
        'cross_test':(('Coverage-nDCG@10','cross_nDCG10'),('FullCoveragePrecision@10','cross_FCP10'),
                      ('FullCoverage_mAP@100','cross_FCmAP100'))}
    result=[]
    for role,left in ordinary.items():
        right=cross[role]
        identity=('roles','model','variant','alpha','beta')
        if any(left[k]!=right[k] for k in identity):
            raise ValueError('Encoder/weight identity differs between protocols')
        row={k:left[k] for k in identity}
        for stage,source,prefix in [('test',left,'ordinary'),('cross_test',right,'cross')]:
            row[prefix+'_source_key']=source['source_key']
            row[prefix+'_source_path']=source['source_path']
            row[prefix+'_historical_reference']=source['historical_reference']
            for old,new in columns[stage]:
                row[new]=source[old]
                row[new+'_effective_n']=source[old+'_effective_n']
        result.append(row)
    return result


def _code():
    return {str(path):file_hash(path) for path in [Path(__file__).resolve(),ROOT/'cnn_augmentation/reporting.py',
      ROOT/'f_region_structure/state.py',ROOT/'wbm_retrieval/common.py']}


def verify_report(output):
    output=Path(output);folder=output/'report';manifest=read_signed(folder/'manifest.json')
    if manifest['context_signature']!=read_signed(output/'run_manifest.json')['signature']:
        raise ValueError('Report context changed')
    if manifest['code']!=_code():raise ValueError('Report code identity changed')
    _check_binding(manifest['source_binding'])
    if not verified(folder,manifest['signature']):raise ValueError('Report is not sealed')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual!=REPORT_FILES:raise ValueError('Unexpected or missing report artifact')
    summary=read_json(folder/'summary.json')
    if summary.get('signature')!=manifest['signature'] or summary.get('context_signature')!=manifest['context_signature']:
        raise ValueError('Report summary detached from manifest')
    audit=read_json(folder/'verification.json')
    if (audit.get('status')!='verified' or audit.get('signature')!=manifest['signature']
            or audit.get('context_signature')!=manifest['context_signature']):
        raise ValueError('Report verification detached from manifest')
    return summary


def run_report(output):
    output=Path(output);folder=output/'report'
    if (folder/'completed.json').exists():return verify_report(output)
    if folder.exists() and any(folder.iterdir()):raise ValueError('Unsealed partial report; preserve it before retry')
    root=read_signed(output/'run_manifest.json');binding={};evaluations={}
    for name in ('run_manifest.json','evaluation_index_valid.json','controls_index_valid.json','evaluation_index_test.json','selection.json','wbbs_references.json'):
        _bind(output/name,binding)
    selection=read_signed(output/'selection.json');wbbs=read_signed(output/'wbbs_references.json')
    if selection['context_signature']!=root['signature'] or wbbs['context_signature']!=root['signature']:
        raise ValueError('Selection/reference context changed')
    if selection['selection_split']!='valid' or selection['test_used'] is not False:
        raise ValueError('Weights must be selected on valid only')
    if selection['evaluation_index_valid_sha256']!=file_hash(output/'evaluation_index_valid.json'):
        raise ValueError('Selection detached from the complete valid index')
    winner=selection['winner']
    if winner not in GRID or selection['theta0']!=THETA0 or selection['spec']!=GRID[winner]:raise ValueError('Unfrozen selection')
    selected=list(dict.fromkeys([THETA0,winner]))
    if selection['selected_variants']!=selected or selection['winner_is_theta0']!=(winner==THETA0):
        raise ValueError('Selected endpoint alias declaration differs')
    _check_binding(wbbs['source_binding']);binding.update(wbbs['source_binding'])
    for name in ('evaluation_index_valid.json','controls_index_valid.json','evaluation_index_test.json'):
        index=read_signed(output/name)
        if index['context_signature']!=root['signature']:raise ValueError('Evaluation index belongs to another context')
        stages=['valid'] if name=='evaluation_index_valid.json' else (['cross_valid'] if name=='controls_index_valid.json' else ['test','cross_test'])
        variants=GRID if stages==['valid'] else selected
        expected={(m,s,v) for m in ('baseline','rotation') for s in stages for v in variants}
        if {(r['model'],r['stage'],r['variant']) for r in index['evaluations'].values()}!=expected:
            raise ValueError('Missing or unauthorized report endpoint roles')
        for row in index['evaluations'].values():
            ident=(row['model'],row['stage'],row['variant'])
            if ident in evaluations and evaluations[ident]['row']!=row:raise ValueError('Conflicting indexed endpoint')
            evaluations[ident]=_load_row(row,binding)
    valid=[]
    for variant in GRID:
        current=evaluations[('rotation','valid',variant)];old=evaluations[('baseline','valid',variant)]
        q=current['families']['exact']['queries'];hit_values=[float(x['Precision@10'])*10 for x in q]
        if any(abs(v-round(v))>1e-8 for v in hit_values):raise ValueError('P10 cannot be converted to integer hits')
        hits=sum(round(v) for v in hit_values)
        valid.append({'variant':variant,**GRID[variant],'top10_hits':hits,'num_queries':len(q),
          'rotation_Precision@10':current['families']['exact']['document']['metrics']['Precision@10'],
          'baseline_Precision@10':old['families']['exact']['document']['metrics']['Precision@10'],
          'rotation_mAP@100':current['families']['exact']['document']['metrics']['mAP@100'],
          'rotation_Coverage-nDCG@10':current['families']['coverage']['document']['metrics']['Coverage-nDCG@10'],
          'rotation_source_key':current['row']['key'],'baseline_source_key':old['row']['key']})
    valid.sort(key=lambda x:(-x['top10_hits'],x['variant']!=THETA0,x['variant']))
    if valid[0]['variant']!=winner:raise ValueError('Selected winner differs from valid-only integer-hit rule')
    if [r['variant'] for r in selection['ranking']]!=[r['variant'] for r in valid]:raise ValueError('Frozen ranking changed')
    for rank,(row,frozen) in enumerate(zip(valid,selection['ranking']),1):
        if (frozen['top10_hits']!=row['top10_hits'] or frozen['num_queries']!=500 or frozen['rank']!=rank
                or abs(frozen['Precision@10']-row['top10_hits']/5000)>1e-12):
            raise ValueError('Frozen valid hit count/rank changed')
        row['rank']=rank;row['selected']=row['variant']==winner
    wbbs_loaded={}
    for role,paths in wbbs['families'].items():
        stage,family=role.split('/');wbbs_loaded[(stage,family)]=_loaded_family(paths,stage,family,binding)
    tables={};aliases={};comparisons=[];pair_checks=[];skipped=[];allmetrics=[]
    for stage in ('test','cross_test'):
        table=[];unique={};aliases[stage]={}
        for role,model,variant in endpoint_roles(THETA0,winner):
            data=evaluations[(model,stage,variant)];row=data['row'];identity=(row['key'],str(Path(row['path']).resolve()))
            aliases[stage][role]={'key':row['key'],'path':row['path'],'model':model,'variant':variant}
            if identity in unique:
                unique[identity]['roles']+=';'+role
                continue
            compact={'roles':role,'model':model,'variant':variant,'alpha':GRID[variant]['alpha'],'beta':GRID[variant]['beta'],
                     'source_key':row['key'],'source_path':row['path'],'historical_reference':row['historical_reference']}
            for family,metric in METRICS[stage]:
                d=data['families'][family]['document'];compact[metric]=d['metrics'][metric];compact[metric+'_effective_n']=d['metric_query_counts'][metric]
            table.append(compact);unique[identity]=compact
        compact={'roles':'WBBS','model':'wbbs','variant':'canonical','alpha':None,'beta':None,
           'source_key':'wbbs_test' if stage=='test' else 'cross_wbbs',
           'source_path':str(Path(wbbs_loaded[(stage,'coverage')]['paths']['metrics']).parent),'historical_reference':True}
        for family,metric in METRICS[stage]:
            d=wbbs_loaded[(stage,family)]['document'];compact[metric]=d['metrics'][metric];compact[metric+'_effective_n']=d['metric_query_counts'][metric]
        table.append(compact);tables[stage]=table
        aliases[stage]['WBBS']={'key':compact['source_key'],'path':compact['source_path'],
            'model':'wbbs','variant':'canonical','families':{
                family:wbbs_loaded[(stage,family)]['paths'] for family,_ in METRICS[stage]}}
        candidate=evaluations[('rotation',stage,winner)]
        for reference_role in ('new_theta0','WBBS'):
            reference=evaluations[('rotation',stage,THETA0)] if reference_role=='new_theta0' else None
            if reference is not None and reference['row']['key']==candidate['row']['key']:
                skipped.append({'stage':stage,'reference':reference_role,'candidate':'new_theta_star','reason':'identical canonical endpoint; no independent comparison or model'})
                continue
            for family,metric in METRICS[stage]:
                rf=reference['families'][family] if reference is not None else wbbs_loaded[(stage,family)]
                cf=candidate['families'][family]
                a,b,check=checked_pair(rf['queries'],cf['queries'],family)
                pair_checks.append({'stage':stage,'reference':reference_role,'metric':metric,**check})
                for aggregation in ('query_mean','class_macro'):
                    ci=paired_query_bootstrap(a,b,per_query_metric(metric),repetitions=2000,seed=2026,
                           aggregation=aggregation,expected_classes=37)
                    comparisons.append({'stage':stage,'family':family,'reference_role':reference_role,'candidate_role':'new_theta_star',
                        'reference_key':reference['row']['key'] if reference is not None else compact['source_key'],
                        'candidate_key':candidate['row']['key'],**ci,'metric':metric,'per_query_metric':per_query_metric(metric)})
    for (model,stage,variant),data in evaluations.items():
        for family,f in data['families'].items():
            allmetrics.extend(_flat(f['document'],{'model':model,'stage':stage,'variant':variant,'family':family,'source_key':data['row']['key']}))
    for (stage,family),f in wbbs_loaded.items():
        allmetrics.extend(_flat(f['document'],{'model':'wbbs','stage':stage,'variant':'canonical','family':family,'source_key':wbbs['families'][stage+'/'+family]['canonical_key']}))
    merged=merge_protocol_tables(tables)
    _check_binding(binding)
    identity={'context_signature':root['signature'],'selection_sha256':file_hash(output/'selection.json'),
      'source_binding':dict(sorted(binding.items())),'code':_code(),'bootstrap':2000,'seed':2026,'scope':SCOPE}
    manifest=signed(identity)
    folder.mkdir(parents=True,exist_ok=True)
    record(folder/'manifest.json',manifest)
    write_csv(folder/'valid_selection.csv',valid,list(valid[0]))
    for stage,table in tables.items():write_csv(folder/('comparison_'+stage+'.csv'),table,list(table[0]))
    write_csv(folder/'comparison_wbbs.csv',merged,list(merged[0]))
    write_csv(folder/'paired_test_comparisons.csv',comparisons,list(comparisons[0]))
    write_csv(folder/'all_metrics.csv',allmetrics,list(allmetrics[0]))
    record(folder/'role_aliases.json',{'theta0':THETA0,'winner':winner,'roles':aliases,'skipped_identical_comparisons':skipped})
    verification={'status':'verified','signature':manifest['signature'],'context_signature':root['signature'],
      'valid_selector':'sum integer top10 hits; ties theta0 then lexical variant; test not read by selection',
      'valid_variants':12,'same_wbbs_ordinary_ranking':wbbs['same_ordinary_ranking'],
      'pairing_checks':pair_checks,'source_writes':0,'normalization':'strict 8-bit or sorted pipe-index labels; per-source class/signature bijection; query-by-query equality of labels/count/pool/relevant denominator; common class IDs from sorted full signatures',
      'source_means_and_effective_counts_recomputed':True}
    record(folder/'verification.json',verification)
    summary={'signature':manifest['signature'],'context_signature':root['signature'],'winner':winner,'spec':GRID[winner],
       'selection_split':'valid','test_used_for_selection':False,'training_required':False,'theta0':THETA0,
       'valid_selection':valid,'comparison_test':tables['test'],'comparison_cross_test':tables['cross_test'],
       'comparison_wbbs':merged,'main_table':'comparison_wbbs.csv',
       'paired_comparisons':comparisons,'role_aliases':aliases,'skipped_identical_comparisons':skipped,
       'limitations':[SCOPE,'Posthoc rotation-model and weight study; valid-only weight choice does not erase prior test exposure.',
       'The 2x2 separates frozen encoder and weight roles; improvement combining both is not a single-factor augmentation effect.',
       'Canonical repeated endpoints are references/aliases, not new models or independent observations.',
       'WBBS uses identical frozen queries, complete eligible gallery and relevance definitions; methods retain their original training and scoring differences.']}
    record(folder/'summary.json',summary)
    _check_binding(binding)
    seal(folder,manifest['signature'])
    return verify_report(output)
