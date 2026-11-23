from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cnn_augmentation.reporting import (MODELS,PLAN,paired_query_bootstrap,
                                      build_report,STYLES)
from cnn_augmentation.state import signed,VERSION
from cnn_augmentation.training import AUDIT_COUNTERS,audit_history
from f_region_structure.state import seal
from wbm_retrieval.common import write_json,write_csv,file_hash,read_json


def rows(values,labels=None,metric='Precision@10'):
    return [{'query_id':i+100,'class_id':i if labels is None else labels[i],
             'label_signature':str(i if labels is None else labels[i]),'label_count':1,
             'pool_size':500,metric:value} for i,value in enumerate(values)]


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
        full=[]
        for row in history:
            one={**row,**{key:0 for key in AUDIT_COUNTERS}}
            n=99786
            one.update(sample_count=n,augmentation_applied_count=n,
                target_before_pixels=n*10,target_after_pixels=n*9,target_retained_source_pixels=n*9,
                target_retention_eligible_count=n,target_retention_fraction_sum=n*.9,
                support_before_pixels=n*100,support_after_pixels=n*90,support_retained_source_pixels=n*90,
                support_retention_eligible_count=n,support_retention_fraction_sum=n*.9)
            full.append(one)
        audit=audit_history(full,name)
        write_json(train/'history.json',full)
        write_json(train/'augmentation_audit.json',audit)
        write_json(train/'training_summary.json',{'completed':True,'augmentation_audit':audit})
        seal(train,'train')
    write_json(root/'augmentation_audit/manifest.json',{'signature':'input-audit'})
    write_json(root/'augmentation_audit/summary.json',{'signature':'input-audit','context_signature':signature})
    seal(root/'augmentation_audit','input-audit')
    for model in MODELS:
        for split in ('valid','test'):
            index={'context_signature':signature,'evaluations':{}}
            for stage in (split,'cross_'+split):
                for style in STYLES:
                    path=root/'source_reports'/model/stage/style
                    history_ref=model=='baseline'
                    value=.5+(0 if model=='baseline' else .02)
                    coverage={'Coverage-nDCG@10':value,'FullCoveragePrecision@10':value,'FullCoverage_mAP@100':value}
                    exact={'Precision@10':value,'mAP@100':value}
                    for family,metrics in (('coverage',coverage),('exact',exact)):
                        if family=='exact' and stage.startswith('cross_'):continue
                        prefix='coverage_' if family=='coverage' else ''
                        qrows=rows([value]*37,metric=next(iter(metrics)))
                        for row in qrows:
                            row.update(metrics)
                            row.update({m.replace('FullCoverage_mAP@','FullCoverageAP@').replace('mAP@','AP@'):v for m,v in metrics.items()})
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

    def test_full_report_seals_48_roles_and_twelve_figure_files(self):
        with tempfile.TemporaryDirectory() as temp:
            report_fixture(temp)
            # Source evaluation validity is tested by evaluation/register tests;
            # this fixture exercises report compilation and all paired joins.
            def validated(output,model,split,root_manifest=None):
                return read_json(Path(output)/'evaluation'/model/('evaluation_index_'+split+'.json'))
            with patch('cnn_augmentation.register.validate_model_split',side_effect=validated):
                summary=build_report(temp)
            self.assertEqual(len(summary['evaluations']),48)
            self.assertEqual(summary['new_evaluations'],40)
            self.assertEqual(summary['historical_references'],8)
            self.assertTrue(summary['decisions']['crop']['main_retrieval_criteria_met'])
            files=[Path(path) for paths in summary['figures'].values() for path in paths]
            self.assertEqual(len(files),12)
            self.assertTrue(all(path.stat().st_size>500 for path in files))
            self.assertTrue((Path(temp)/'report/paper_main_metrics.csv').is_file())
            self.assertEqual(len([r for r in summary['all_metrics'] if r['group']=='class']),37*192)
            with patch('cnn_augmentation.register.validate_model_split',side_effect=validated):
                self.assertEqual(build_report(temp)['signature'],summary['signature'])


if __name__=='__main__':unittest.main()
