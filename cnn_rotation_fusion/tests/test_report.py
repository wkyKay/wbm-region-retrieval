import copy
import json
from pathlib import Path

import pytest

from cnn_rotation_fusion import report
from f_region_structure.state import record, seal, signed
from wbm_retrieval.common import file_hash


def query(q,cls,label,count=2,pool=10,relevant=3):
    return {'query_id':str(q),'class_id':str(cls),'label_signature':label,
            'label_count':str(count),'pool_size':str(pool),'relevant_in_pool':str(relevant),
            'full_coverage_in_pool':str(relevant),'Precision@10':'0.2','AP@100':'0.1',
            'Coverage-nDCG@10':'0.5','FullCoverageAP@100':'0.05'}


def test_pipe_and_binary_encodings_join_without_mutation():
    a=[query(2,0,'0|2'),query(4,1,'1|3')]
    b=[query(4,9,'01010000'),query(2,12,'10100000')]
    old=copy.deepcopy((a,b))
    left,right,check=report.checked_pair(a,b,'exact')
    assert (a,b)==old
    assert {int(x['query_id']):x['class_id'] for x in left}=={int(x['query_id']):x['class_id'] for x in right}
    assert check['all_query_labels_pools_and_denominators_equal']
    assert check['reference_encoding']['source_encodings']=={'pipe_indices':2}
    assert check['candidate_encoding']['source_encodings']=={'binary_8':2}


@pytest.mark.parametrize('rows',[
    [query(2,0,'0|2'),query(4,0,'1|3')],
    [query(2,0,'0|2'),query(4,1,'0|2')],
    [query(2,0,'0|2'),query(2,0,'0|2')],
    [query(2,0,'0|2',count=3)],
    [query(2,0,'00000000',count=0)],
    [query(2,0,'2|0')],
    [query(2,0,'0|0')],
])
def test_nonbijective_duplicate_or_invalid_labels_fail(rows):
    with pytest.raises(ValueError):report.normalize_queries(rows)


@pytest.mark.parametrize('key,value',[
    ('query_id','99'),('pool_size','11'),('relevant_in_pool','4'),
    ('label_signature','01010000'),
])
def test_pairing_rejects_query_label_pool_and_relevance_changes(key,value):
    a=[query(2,1,'0|2')];b=[query(2,7,'10100000')]
    b[0][key]=value
    with pytest.raises(ValueError):report.checked_pair(a,b,'exact')


def test_coverage_full_denominator_cannot_be_ignored():
    a=[query(2,1,'0|2')];b=[query(2,7,'10100000')]
    b[0]['full_coverage_in_pool']='0'
    with pytest.raises(ValueError,match='denominator'):report.checked_pair(a,b,'coverage')


def test_missing_coverage_values_remain_undefined():
    a=[query(2,1,'0|2',relevant=0),query(4,2,'1|3')]
    b=[query(2,9,'10100000',relevant=0),query(4,7,'01010000')]
    a[0]['FullCoverageAP@100']=b[0]['FullCoverageAP@100']=''
    b[1]['FullCoverageAP@100']='0.1'
    left,right,check=report.checked_pair(a,b,'coverage')
    result=report.paired_query_bootstrap(left,right,'FullCoverageAP@100',repetitions=2000,seed=2026,expected_classes=2)
    assert result['total_queries']==2 and result['paired_effective_n']==1
    assert result['excluded_nonfinite_pairs']==1
    assert result['difference']==pytest.approx(.05)
    assert result['ci_low']==result['ci_high']==pytest.approx(.05)


def test_theta0_winner_roles_are_explicit_aliases():
    roles=report.endpoint_roles(report.THETA0,report.THETA0)
    assert len(roles)==4
    assert len({(model,variant) for _,model,variant in roles})==2
    other=next(k for k in report.GRID if k!=report.THETA0)
    assert len({(model,variant) for _,model,variant in report.endpoint_roles(report.THETA0,other)})==4


def test_family_requires_sealed_report_and_pinned_manifest(tmp_path):
    folder=tmp_path/'test';folder.mkdir()
    meta=signed({'kind':'fixture'});record(tmp_path/'evaluation_manifest.json',meta)
    for fn in ['metrics.json','per_query.csv','per_class.csv']:(folder/fn).write_text('{}')
    seal(tmp_path,meta['signature'])
    binding={}
    got=report._family(tmp_path,'test','exact',binding,file_hash(tmp_path/'completed.json'),file_hash(tmp_path/'evaluation_manifest.json'))
    assert got['metrics']==str(folder/'metrics.json')
    (folder/'per_query.csv').write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):report._family(tmp_path,'test','exact',{})


def test_external_source_change_is_rejected(tmp_path):
    p=tmp_path/'source.csv';p.write_text('original')
    binding={};report._bind(p,binding)
    p.write_text('modified')
    with pytest.raises(ValueError,match='source changed'):report._check_binding(binding)


def protocol_tables(alias=False):
    roles=['old_theta0','new_theta0','old_theta_star','new_theta_star','WBBS']
    if alias:roles=['old_theta0;old_theta_star','new_theta0;new_theta_star','WBBS']
    result={'test':[],'cross_test':[]}
    for stage in result:
        for role in roles:
            item={'roles':role,'model':'wbbs' if role=='WBBS' else role.split('_')[0],
                  'variant':'theta0' if alias else role,'alpha':None,'beta':None,
                  'source_key':stage+'_'+role,'source_path':'/'+stage+'/'+role,'historical_reference':True}
            for metric in ['Precision@10','mAP@100','Coverage-nDCG@10','FullCoveragePrecision@10','FullCoverage_mAP@100']:
                item[metric]=.1 if stage=='test' else .2
                item[metric+'_effective_n']=7199 if stage=='test' else {'Coverage-nDCG@10':7000,'FullCoverage_mAP@100':6200}.get(metric,7199)
            result[stage].append(item)
    return result


@pytest.mark.parametrize('alias,expected',[(False,5),(True,3)])
def test_single_wbbs_table_keeps_protocols_counts_and_aliases(alias,expected):
    tables=protocol_tables(alias);before=copy.deepcopy(tables)
    merged=report.merge_protocol_tables(tables)
    assert tables==before and len(merged)==expected
    assert merged[-1]['roles']=='WBBS'
    for row in merged:
        assert row['ordinary_exact_P10']==.1 and row['cross_nDCG10']==.2
        assert row['ordinary_exact_mAP100_effective_n']==7199
        assert row['cross_nDCG10_effective_n']==7000
        assert row['cross_FCP10_effective_n']==7199
        assert row['cross_FCmAP100_effective_n']==6200
    if alias:assert merged[0]['roles']=='old_theta0;old_theta_star'


@pytest.mark.parametrize('mutation',['missing','duplicate','identity'])
def test_combined_table_refuses_mismatched_protocol_roles(mutation):
    tables=protocol_tables()
    if mutation=='missing':tables['cross_test'].pop()
    elif mutation=='duplicate':tables['cross_test'].append(dict(tables['cross_test'][0]))
    else:tables['cross_test'][0]['variant']='wrong_weight'
    with pytest.raises(ValueError):report.merge_protocol_tables(tables)


def test_combined_table_is_required_in_exact_report_seal_inventory():
    assert 'comparison_wbbs.csv' in report.REPORT_FILES
    assert {'comparison_test.csv','comparison_cross_test.csv'}<=report.REPORT_FILES
    assert len(report.REPORT_FILES)==11
