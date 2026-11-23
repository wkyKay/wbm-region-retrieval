from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cnn_refinement.reporting import (MODELS,PLAN,paired_query_bootstrap,context_statistics,
                                      build_report,STYLES)
from cnn_refinement.state import signed,VERSION
from f_region_structure.state import seal
from wbm_retrieval.common import write_json,write_csv,file_hash,read_json


def rows(values,labels=None,metric='Precision@10'):
    return [{'query_id':i+100,'class_id':i if labels is None else labels[i],
             'label_signature':str(i if labels is None else labels[i]),'label_count':1,
             'pool_size':500,metric:value} for i,value in enumerate(values)]


def context_rows(effect=0.):
    result=[]
    for sid in range(4):
        for view in ('context_drop','target_only','rot90'):
            result.append({'space':'h512','view':view,'map_id':sid,'region_id':1,'label_count':2,
                'token_index':sid,'label_signature':str(sid%2),'has_context':True,'is_noop':False,
                'valid_region_input':True,'source_all_defect_changed_pixels':4,
                'cosine_distance':(.2 if view=='context_drop' else .4)-effect})
    return result


def report_fixture(root):
    root=Path(root)
    baseline=root/'old_training';baseline.mkdir()
    history=[{'epoch':i,'train_loss':2/i,'valid_nce':1/i} for i in range(1,101)]
    write_json(baseline/'history.json',history)
    plan=signed({'plan':PLAN})
    manifest=signed({'version':VERSION,'plan_signature':plan['signature'],'baseline':{'checkpoint':str(baseline/'best_nce.pt')}})
    signature=manifest['signature']
    write_json(root/'run_manifest.json',manifest)
    write_json(root/'frozen_plan.json',plan)
    for name in MODELS[1:]:
        train=root/'training'/name
        write_json(train/'history.json',history);write_json(train/'training_summary.json',{'completed':True})
        seal(train,'train')
    geometry={}
    for name in MODELS:
        effect=0 if name=='baseline' else .03
        write_json(root/'diagnostics'/name/'context_rows.json',context_rows(effect))
        geometry[name+'/valid_representative/h512']={'class_balanced':{'metrics':{'hit_at_1':.5+effect,'precision_at_10':.5+effect}},
            'dispersion':{'zero_normalized_variance':False,'covariance_effective_rank':15,
                          'normalized_mean_dimension_variance':.03}}
    write_json(root/'diagnostics/summary.json',{'geometry':geometry})
    seal(root/'diagnostics','diagnostics')
    write_json(root/'negative_diagnostics/summary.json',{'single_label_proxy':{'precision_micro':.6},
              'observed_diagnostic_sample':{'replacement_fraction':.004}})
    seal(root/'negative_diagnostics','negative')
    for model in MODELS:
        for split in ('valid','test'):
            index={'context_signature':signature,'evaluations':{}}
            for stage in (split,'cross_'+split):
                for style in STYLES:
                    path=root/'source_reports'/model/stage/style
                    history_ref=model=='baseline' and style=='fusion'
                    value=.5+(0 if model=='baseline' else .02)
                    coverage={'Coverage-nDCG@10':value,'FullCoveragePrecision@10':value}
                    exact={'Precision@10':value}
                    for family,metrics in (('coverage',coverage),('exact',exact)):
                        if family=='exact' and stage.startswith('cross_'):continue
                        prefix='coverage_' if family=='coverage' else ''
                        qrows=rows([value]*37,metric=next(iter(metrics)))
                        for row in qrows:row.update(metrics)
                        aggregate={'metrics':metrics,'metric_query_counts':{m:37 for m in metrics}}
                        report={**aggregate,'class_macro_metrics':metrics,
                            'class_macro_metric_class_counts':{m:37 for m in metrics},
                            'groups':{'single_label':aggregate,'mixed_label':aggregate,'by_label_count':{'1':aggregate}}}
                        write_json(path/(prefix+'metrics.json'),report)
                        write_csv(path/(prefix+'per_query.csv'),qrows,list(qrows[0]))
                        write_csv(path/(prefix+'per_class.csv'),qrows,list(qrows[0]))
                    write_json(path/'evaluation_manifest.json',{'signature':'evaluation'})
                    write_json(path/'verification.json',{'status':'verified','all_queries_full_candidate_topk_exact':True})
                    seal(path,'evaluation')
                    item={'path':str(path),'model':model,'style':style,'stage':stage,'split':split,
                          'key':'_'.join((model,stage,style)),'register':not history_ref,'historical_reference':history_ref,
                          'completed_sha256':file_hash(path/'completed.json')}
                    index['evaluations'][stage+'/'+style]=item
            write_json(root/'evaluation'/model/('evaluation_index_'+split+'.json'),index)


class ReportingTests(unittest.TestCase):
    def test_query_join_handles_order_and_nonfinite_common_ids(self):
        a=rows([.1,None,.3,.5],[0,0,1,1]);b=rows([.2,.3,float('nan'),.6],[0,0,1,1])
        result=paired_query_bootstrap(a,b[::-1],'Precision@10',repetitions=100,expected_classes=2)
        self.assertEqual(result['paired_effective_n'],2)
        self.assertEqual(result['reference_effective_n'],3)
        self.assertEqual(result['candidate_effective_n'],3)
        self.assertAlmostEqual(result['difference'],.1)
        self.assertAlmostEqual(result['ci_low'],.1)
        self.assertAlmostEqual(result['ci_high'],.1)
        with self.assertRaisesRegex(ValueError,'identical full query'):
            paired_query_bootstrap(a,b[:-1],'Precision@10',repetitions=100)

    def test_query_weighted_and_macro_bootstrap_differ_with_unbalanced_classes(self):
        a=rows([0,0,0,0],[0,0,0,1]);b=rows([0,0,0,1],[0,0,0,1])
        query=paired_query_bootstrap(a,b,'Precision@10',repetitions=100)
        macro=paired_query_bootstrap(a,b,'Precision@10',repetitions=100,aggregation='class_macro')
        self.assertEqual(query['difference'],.25)
        self.assertEqual(macro['difference'],.5)
        changed=deepcopy(b);changed[0]['pool_size']=499
        with self.assertRaisesRegex(ValueError,'candidate-pool'):
            paired_query_bootstrap(a,changed,'Precision@10',repetitions=100)

    def test_context_distances_and_relative_effects_use_paired_wafer_cohort(self):
        data={name:context_rows(0 if name=='baseline' else .1) for name in MODELS}
        result=context_statistics(data,repetitions=100)
        selected=next(r for r in result if r['model']=='context_dropout' and r['view']=='context_drop')
        self.assertEqual(selected['maps'],4)
        self.assertAlmostEqual(selected['relative_reduction'],.5)
        self.assertAlmostEqual(selected['relative_reduction_ci_low'],.5)

    def test_full_report_seals_24_results_and_twelve_figure_files(self):
        with tempfile.TemporaryDirectory() as temp:
            report_fixture(temp)
            # Source evaluation validity is tested by evaluation/register tests;
            # this fixture exercises report compilation and all paired joins.
            def validated(output,model,split,root_manifest=None):
                return read_json(Path(output)/'evaluation'/model/('evaluation_index_'+split+'.json'))
            with patch('cnn_refinement.register.validate_model_split',side_effect=validated):
                summary=build_report(temp)
            self.assertEqual(len(summary['evaluations']),24)
            self.assertEqual(summary['new_evaluations'],20)
            self.assertEqual(summary['historical_references'],4)
            self.assertTrue(summary['decisions']['context_dropout']['main_retrieval_criteria_met'])
            files=[Path(path) for paths in summary['figures'].values() for path in paths]
            self.assertEqual(len(files),12)
            self.assertTrue(all(path.stat().st_size>500 for path in files))
            self.assertTrue((Path(temp)/'report/paper_main_metrics.csv').is_file())
            self.assertEqual(len([r for r in summary['all_metrics'] if r['group']=='class']),37*60)
            with patch('cnn_refinement.register.validate_model_split',side_effect=validated):
                self.assertEqual(build_report(temp)['signature'],summary['signature'])


if __name__=='__main__':unittest.main()
