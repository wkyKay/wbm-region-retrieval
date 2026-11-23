"""Paper-ready, paired reporting of frozen CNN refinement experiments.

All comparisons join original query IDs, retain complete-combination strata,
and resample paired queries within strata. Intervals condition on this model
seed and fixed gallery; they are not training-seed or gallery uncertainty.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from .state import MODELS, PLAN, run_context
from f_region_structure.state import record, seal, verified
from wbm_retrieval.common import digest, file_hash, read_json, write_csv

STAGES=('valid','cross_valid','test','cross_test')
STYLES=('cnn_only','fusion')
COLORS=dict(zip(MODELS, ('#4b5563','#0072b2','#d55e00','#009e73','#cc79a7','#e69f00')))
LABELS={'baseline':'Baseline (right-angle rotation)','crop':'Crop','cutout':'Cutout','noise':'Noise','rotation':'Arbitrary rotation','shift':'Shift'}
SCOPE='Paired query bootstrap within complete wafer-label combinations; fixed trained model, fixed gallery and this single training seed; not cross-seed uncertainty or a multiple-comparison correction.'


def _finite(value):
    if value is None or value=='':return None
    number=float(value)
    return number if np.isfinite(number) else None


def _rows(path):
    with Path(path).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    ids=[int(row['query_id']) for row in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate original query ID in report')
    return rows


def _paired_arrays(labels,left,right,repetitions,seed,aggregation):
    labels=np.asarray(labels);left=np.asarray(left,dtype=float);right=np.asarray(right,dtype=float)
    if (not len(left) or labels.shape!=left.shape or right.shape!=left.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or aggregation not in ('query_mean','class_macro')):
        raise ValueError('Invalid paired finite values or aggregation')
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(labels==label) for label in sorted(set(labels.tolist()))]
    draws=np.zeros((repetitions,2));means=np.zeros(2)
    for indices in groups:
        values=np.stack((left[indices],right[indices]),1)
        weight=len(indices)/len(left) if aggregation=='query_mean' else 1/len(groups)
        means+=weight*values.mean(0)
        for start in range(0,repetitions,128):
            count=min(128,repetitions-start)
            sampled=rng.integers(len(indices),size=(count,len(indices)))
            draws[start:start+count]+=weight*values[sampled].mean(1)
    return means,draws


def paired_query_bootstrap(reference,candidate,metric,*,repetitions=2000,seed=2026,
                           aggregation='query_mean',expected_classes=None):
    """Join all query IDs, then exclude only nonfinite metric pairs transparently."""
    if type(repetitions) is not int or repetitions<1:raise ValueError('Invalid bootstrap repetitions')
    def indexed(rows):
        result={int(row['query_id']):row for row in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate query identity')
        return result
    left,right=indexed(reference),indexed(candidate)
    if not left or left.keys()!=right.keys():raise ValueError('Paired reports must contain identical full query IDs')
    labels=[];a=[];b=[];ids=[];nleft=nright=0;all_classes=set()
    for q in sorted(left):
        x,y=left[q],right[q]
        if (x['label_signature']!=y['label_signature'] or int(x['class_id'])!=int(y['class_id'])
                or int(x['label_count'])!=int(y['label_count']) or int(x['pool_size'])!=int(y['pool_size'])):
            raise ValueError('Paired query labels or complete candidate-pool size changed')
        all_classes.add(x['label_signature'])
        xv,yv=_finite(x[metric]),_finite(y[metric])
        nleft+=xv is not None;nright+=yv is not None
        if xv is not None and yv is not None:
            ids.append(q);labels.append(x['label_signature']);a.append(xv);b.append(yv)
    if expected_classes is not None and len(all_classes)!=expected_classes:
        raise ValueError('Full query population lost a complete-combination stratum')
    result={'metric':metric,'aggregation':aggregation,'total_queries':len(left),
        'reference_effective_n':nleft,'candidate_effective_n':nright,'paired_effective_n':len(ids),
        'full_classes':len(all_classes),'effective_classes':len(set(labels)),
        'excluded_nonfinite_pairs':len(left)-len(ids),'paired_query_ids_sha256':digest(ids),
        'repetitions':repetitions,'seed':seed,'confidence':.95,'scope':SCOPE}
    if not ids:return {**result,'reference':None,'candidate':None,'difference':None,'ci_low':None,'ci_high':None}
    means,draws=_paired_arrays(labels,a,b,repetitions,seed,aggregation)
    low,high=np.quantile(draws[:,1]-draws[:,0],[.025,.975])
    return {**result,'reference':float(means[0]),'candidate':float(means[1]),
            'difference':float(means[1]-means[0]),'ci_low':float(low),'ci_high':float(high)}


def _bind(path,binding):
    path=Path(path).resolve()
    sha=file_hash(path)
    if str(path) in binding and binding[str(path)]!=sha:raise ValueError('Report source changed')
    binding[str(path)]=sha
    return path


def _sealed_files(folder,names,binding,expected_marker=None):
    folder=Path(folder)
    marker=read_json(_bind(folder/'completed.json',binding))
    if marker.get('status')!='completed':raise ValueError('Report source is incomplete')
    if expected_marker and file_hash(folder/'completed.json')!=expected_marker:
        raise ValueError('Evaluation index detached from its sealed marker')
    for name in names:
        path=_bind(folder/name,binding)
        if marker['files'].get(name)!=file_hash(path):raise ValueError('Report source no longer matches its seal: '+str(path))
    return marker


def _flatten_report(document,model,stage,style,family,key):
    base={'model':model,'stage':stage,'split':stage.removeprefix('cross_'),'style':style,'family':family,'source_key':key,
          'class_id':None,'label_signature':None,'label_count':None}
    rows=[]
    groups={'overall':document,'class_macro':{'metrics':document['class_macro_metrics'],
            'metric_query_counts':document['metric_query_counts'],
            'metric_class_counts':document['class_macro_metric_class_counts']},
            'single_label':document['groups']['single_label'],'mixed_label':document['groups']['mixed_label']}
    groups.update({'label_count_'+k:v for k,v in document['groups']['by_label_count'].items()})
    for group,summary in groups.items():
        for metric,value in summary['metrics'].items():
            rows.append({**base,'group':group,'metric':metric,'value':value,
                'effective_n':summary.get('metric_query_counts',{}).get(metric),
                'effective_classes':summary.get('metric_class_counts',{}).get(metric)})
    return rows


def _class_rows(path,queries,document,model,stage,style,family,key):
    """Preserve each class's metric mean and derive its actual finite-query N."""
    with Path(path).open(newline='') as stream:saved=list(csv.DictReader(stream))
    groups=defaultdict(list)
    for row in queries:groups[int(row['class_id'])].append(row)
    if len(saved)!=len(groups) or {int(row['class_id']) for row in saved}!=set(groups):
        raise ValueError('Class report does not match original query classes')
    result=[]
    for row in saved:
        cls=int(row['class_id']);members=groups[cls]
        for metric in document['metrics']:
            field=metric.replace('FullCoverage_mAP@','FullCoverageAP@') if metric.startswith('FullCoverage_mAP@') else metric
            if field.startswith('mAP@'):field=field.replace('mAP@','AP@',1)
            values=[_finite(query[field]) for query in members]
            values=[value for value in values if value is not None]
            value=_finite(row[metric]);mean=float(np.mean(values)) if values else None
            if (value is None)!=(mean is None) or (value is not None and abs(value-mean)>2e-10):
                raise ValueError('Class metric differs from independently grouped queries')
            result.append({'model':model,'stage':stage,'split':stage.removeprefix('cross_'),'style':style,
                'family':family,'source_key':key,'class_id':cls,'label_signature':row['label_signature'],
                'label_count':int(row['label_count']),'group':'class','metric':metric,'value':value,
                'effective_n':len(values),'effective_classes':1 if values else 0})
    return result


def _scalar_table(value,prefix=''):
    rows=[]
    if isinstance(value,dict):
        for name,item in value.items():rows+=_scalar_table(item,prefix+'/'+str(name) if prefix else str(name))
    elif value is None or isinstance(value,(int,float,bool,str)):
        rows.append({'metric':prefix,'value':value})
    return rows


def decisions(comparisons):
    rules=PLAN['decision_rules'];result={}
    lookup={(row['model'],row['stage'],row['style'],row['metric'],row['aggregation']):row for row in comparisons}
    for model in MODELS[1:]:
        exact=lookup[(model,'test','fusion','Precision@10','query_mean')]
        cross=lookup[(model,'cross_test','fusion','Coverage-nDCG@10','query_mean')]
        cnn=lookup[(model,'test','cnn_only','Precision@10','query_mean')]
        improves=(exact['difference'] is not None and exact['difference']>=rules['fusion_exact_P10_absolute_improvement']
                  and exact['ci_low']>rules['paired_95CI_lower_above'])
        preserves=cross['ci_low'] is not None and cross['ci_low']>rules['fusion_cross_nDCG10_noninferiority_lower']
        cnn_improves=cnn['difference'] is not None and cnn['difference']>0
        result[model]={'test_fusion_exact_P10':exact,'test_fusion_cross_nDCG10':cross,
            'test_cnn_only_exact_P10':cnn,'fusion_exact_P10_criterion_met':bool(improves),
            'fusion_cross_nDCG10_noninferiority_met':bool(preserves),
            'cnn_only_P10_observed_improvement':bool(cnn_improves),
            'cnn_only_P10_positive_CI':cnn['ci_low'] is not None and cnn['ci_low']>0,
            'main_retrieval_criteria_met':bool(improves and preserves and cnn_improves),
            'interpretation':'Prespecified descriptive decision for one seed, fixed models and gallery; all five variants retained; no model replacement or checkpoint reselection follows from test ranking.'}
    return result


def _save_figure(figure,folder,stem):
    paths=[]
    for suffix in ('png','pdf','svg'):
        path=folder/(stem+'.'+suffix)
        figure.savefig(path,dpi=240,bbox_inches='tight')
        paths.append(str(path.resolve()))
    return paths


def render_figures(folder,histories,audits,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    result={}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,history in histories.items():
        for ax,metric in zip(axes,('train_loss','valid_nce')):
            ax.plot([r['epoch'] for r in history],[r[metric] for r in history],color=COLORS[name],
                    label=LABELS[name],linewidth=1.4)
            ax.set_xlabel('Epoch');ax.set_ylabel('NCE loss');ax.grid(alpha=.15)
    axes[0].set_title('Training: each single augmentation')
    axes[1].set_title('Common original rotation validation NCE')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.015,'Different positive views change training difficulty; loss alone does not rank retrieval quality.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.86));result['training_nce']=_save_figure(fig,folder,'training_nce');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    names=list(MODELS[1:]);x=np.arange(len(names))
    for ax,metric,title in zip(axes,('applied_fraction','mean_target_retention_fraction','newly_empty_target_fraction'),
            ('Augmentation applied','Original target pixels retained','Target erased completely')):
        values=[audits[n]['rates'][metric]*100 for n in names]
        ax.bar(x,values,color=[COLORS[n] for n in names],width=.65)
        ax.set_xticks(x,[LABELS[n] for n in names],rotation=25,ha='right',fontsize=8)
        ax.set_ylabel('Percent of presentations' if metric!='mean_target_retention_fraction' else 'Mean percent of source target pixels')
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15)
        ax.set_ylim(0,max(max(values)*1.18,1))
        for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Actual exposure over all 100 epochs; source-pixel retention excludes initially empty targets',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));result['augmentation_exposure']=_save_figure(fig,folder,'augmentation_exposure');plt.close(fig)
    for split in ('valid','test'):
        fig,axes=plt.subplots(2,3,figsize=(14,7),sharey=True)
        panels=((split,'Precision@10','Ordinary exact P@10'),
                ('cross_'+split,'Coverage-nDCG@10','Cross-combination nDCG@10'),
                ('cross_'+split,'FullCoveragePrecision@10','Cross-combination full-coverage P@10'))
        for i,style in enumerate(STYLES):
            for ax,(stage,metric,title) in zip(axes[i],panels):
                rows=[next(r for r in comparisons if r['model']==name and r['stage']==stage
                      and r['style']==style and r['metric']==metric and r['aggregation']=='query_mean') for name in names]
                for j,(name,row) in enumerate(zip(names,rows)):
                    lo,hi,y=[row[k]*100 for k in ('ci_low','ci_high','difference')]
                    ax.errorbar((lo+hi)/2,j,xerr=(hi-lo)/2,fmt='none',capsize=4,color=COLORS[name])
                    ax.plot(y,j,'o',color=COLORS[name])
                ax.axvline(0,color='#777',linewidth=1);ax.grid(axis='x',alpha=.15)
                ax.set_yticks(np.arange(len(names)),[LABELS[n] for n in names]);ax.set_ylim(len(names)-.5,-.5)
                ax.set_xlabel('Change vs baseline (percentage points)',fontsize=9)
                row=rows[0]
                ax.set_title(title+'\n'+('CNN only' if style=='cnn_only' else 'Fixed fusion')+
                             f'; baseline={row["reference"]*100:.2f}%; N={row["paired_effective_n"]}',fontsize=9)
        fig.suptitle(split.capitalize()+': paired query 95% intervals; fixed model, fixed gallery, one seed',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));result['retrieval_'+split]=_save_figure(fig,folder,'retrieval_'+split);plt.close(fig)
    return result


def build_report(output,bootstrap=2000):
    """Require all 48 evaluation roles and complete augmentation exposure, then seal a report."""
    if bootstrap!=2000:raise ValueError('The frozen reporting plan requires 2000 paired bootstrap repetitions')
    output=Path(output).resolve();folder=output/'report';binding={}
    manifest=run_context(output)
    _bind(output/'run_manifest.json',binding)
    plan=read_json(_bind(output/'frozen_plan.json',binding))
    if plan.get('plan')!=PLAN:raise ValueError('Reporting decision rules differ from frozen plan')
    allrows=[];evaluations={};perquery={};new=0;replay=0;main_rows=[]
    from .register import validate_model_split
    for model in MODELS:
        for split in ('valid','test'):
            index=validate_model_split(output,model,split,root_manifest=manifest)
            _bind(output/'evaluation'/model/('evaluation_index_'+split+'.json'),binding)
            if index['context_signature']!=manifest['signature']:raise ValueError('Evaluation index belongs to another experiment')
            for stage in (split,'cross_'+split):
                for style in STYLES:
                    item=index['evaluations'][stage+'/'+style];source=Path(item['path'])
                    if item['model']!=model or item['style']!=style or item['stage']!=stage:
                        raise ValueError('Evaluation index model/style/stage mismatch')
                    new+=bool(item['register']);replay+=bool(item['historical_reference'])
                    names=['evaluation_manifest.json','verification.json','coverage_metrics.json','coverage_per_query.csv','coverage_per_class.csv']
                    if not stage.startswith('cross_'):names+=['metrics.json','per_query.csv','per_class.csv']
                    _sealed_files(source,names,binding,item.get('completed_sha256',item.get('source_completed_sha256')))
                    check=read_json(source/'verification.json')
                    if check.get('status')!='verified' or not check.get('all_queries_full_candidate_topk_exact'):
                        raise ValueError('Source ranking lacks complete-candidate verification')
                    documents={'coverage':read_json(source/'coverage_metrics.json')}
                    perquery[(model,stage,style,'coverage')]=_rows(source/'coverage_per_query.csv')
                    if not stage.startswith('cross_'):
                        documents['exact']=read_json(source/'metrics.json')
                        perquery[(model,stage,style,'exact')]=_rows(source/'per_query.csv')
                    evaluations['/'.join((model,stage,style))]={'source':str(source),'key':item['key'],
                        'historical_reference':item['historical_reference'],'reports':documents}
                    for family,document in documents.items():
                        allrows+=_flatten_report(document,model,stage,style,family,item['key'])
                        filename='coverage_per_class.csv' if family=='coverage' else 'per_class.csv'
                        allrows+=_class_rows(source/filename,perquery[(model,stage,style,family)],document,
                                            model,stage,style,family,item['key'])
                    compact={'model':model,'stage':stage,'style':style,'source_key':item['key']}
                    for family,metric,column in (('exact','Precision@10','exact_P10'),('exact','mAP@100','exact_mAP100'),
                        ('coverage','Coverage-nDCG@10','coverage_nDCG10'),('coverage','FullCoveragePrecision@10','full_coverage_P10'),
                        ('coverage','FullCoverageRecall@10','full_coverage_R10'),('coverage','FullCoverage_mAP@100','full_coverage_mAP100')):
                        document=documents.get(family,{})
                        compact[column]=document.get('metrics',{}).get(metric)
                        compact[column+'_effective_n']=document.get('metric_query_counts',{}).get(metric)
                    main_rows.append(compact)
    if (len(evaluations),new,replay)!=(48,40,8):raise ValueError('Expected 48 roles: 40 new and eight canonical baseline references')
    histories={}
    baseline_history=Path(manifest['baseline']['checkpoint']).parent/'history.json'
    for name in MODELS:
        path=baseline_history if name=='baseline' else output/'training'/name/'history.json'
        if name!='baseline':_sealed_files(path.parent,['history.json','training_summary.json'],binding)
        history=read_json(_bind(path,binding))
        if [r['epoch'] for r in history]!=list(range(1,101)) or any(_finite(r[m]) is None for r in history for m in ('train_loss','valid_nce')):
            raise ValueError('Training history must contain all 100 completed epochs')
        histories[name]=history
    audits={}
    from .register import _augmentation_audit
    for model in MODELS[1:]:
        _sealed_files(output/'training'/model,['augmentation_audit.json'],binding)
        audits[model]=_augmentation_audit(output/'training'/model,model)
    _sealed_files(output/'augmentation_audit',['manifest.json','summary.json'],binding)
    input_audit=read_json(output/'augmentation_audit/summary.json')
    if input_audit.get('context_signature') != manifest['signature']:
        raise ValueError('Input augmentation audit belongs to another run')
    code={str(Path(__file__).with_name(name).resolve()):file_hash(Path(__file__).with_name(name))
          for name in ('reporting.py','state.py','register.py')}
    science={'context_signature':manifest['signature'],'source_binding':dict(sorted(binding.items())),
             'code':code,'bootstrap':bootstrap,'seed':2026,'decision_rules':PLAN['decision_rules'],
             'new_evaluations':new,'historical_references':replay,'scope':SCOPE}
    signature=digest(science)
    if verified(folder,signature):return read_json(folder/'summary.json')
    comparisons=[]
    for model in MODELS[1:]:
        for stage in STAGES:
            targets=[('coverage','Coverage-nDCG@10'),('coverage','FullCoveragePrecision@10'),
                     ('coverage','FullCoverageAP@100')]
            if not stage.startswith('cross_'):targets=[('exact','Precision@10'),('exact','AP@100')]+targets
            for style in STYLES:
                for family,metric in targets:
                    for aggregate in ('query_mean','class_macro'):
                        comparison=paired_query_bootstrap(perquery[('baseline',stage,style,family)],
                            perquery[(model,stage,style,family)],metric,repetitions=bootstrap,
                            aggregation=aggregate,expected_classes=37)
                        display_metric=metric.replace('FullCoverageAP@','FullCoverage_mAP@')
                        if display_metric.startswith('AP@'):display_metric='m'+display_metric
                        comparisons.append({'model':model,'stage':stage,'style':style,'family':family,
                            **comparison,'metric':display_metric,'per_query_metric':metric})
    conclusion=decisions(comparisons)
    folder.mkdir(parents=True,exist_ok=True)
    record(folder/'manifest.json',{**science,'signature':signature})
    write_csv(folder/'all_metrics.csv',allrows,list(allrows[0]))
    write_csv(folder/'paper_main_metrics.csv',main_rows,list(main_rows[0]))
    write_csv(folder/'paired_retrieval_differences.csv',comparisons,list(comparisons[0]))
    historyrows=[{'model':name,**row} for name,history in histories.items() for row in history]
    fields=list(dict.fromkeys(key for row in historyrows for key in row))
    write_csv(folder/'training_history.csv',historyrows,fields)
    auditrows=[{'model':model,'samples':audit['samples'],**audit['rates'],**audit['totals']}
               for model,audit in audits.items()]
    write_csv(folder/'augmentation_audit.csv',auditrows,list(auditrows[0]))
    figures=render_figures(folder/'figures',histories,audits,comparisons)
    display={'main_text':[
        {'figure':'retrieval_test','files':figures['retrieval_test'],
         'metrics':['ordinary exact Precision@10','cross Coverage-nDCG@10','cross FullCoveragePrecision@10'],
         'required_annotation':'All five variants, baseline absolute values, conditional paired95%CI and metric-specific effectiveN; CNN-only and fusion separately'},
        {'table':'single_augmentation_comparison','source':str((folder/'paper_main_metrics.csv').resolve()),
         'metrics':['ordinary exact P@10 and mAP@100','coverage nDCG@10','FullCoverage P@10 and mAP@100'],
         'required_annotation':'ordinary and cross protocols in separate tables; fixed epoch100; single seed; author code adaptations specified'}],
        'appendix':[
            {'figure':'augmentation_examples','source':str((output/'augmentation_audit/summary.json').resolve()),
             'note':'Deterministic pretraining input cohort; original versus each single augmentation and channel semantics.'},
            {'figure':'training_nce','files':figures['training_nce'],'note':'Different augmentation objectives; curves do not determine the best retrieval model.'},
            {'figure':'augmentation_exposure','files':figures['augmentation_exposure'],
             'note':'Actual application, no-op and target erasure/retention counters; target ID retention avoids crop enlargement bias.'},
            {'figure':'retrieval_valid','files':figures['retrieval_valid'],'note':'Not used to select epochs or drop variants.'},
            {'table':'all_seven_K_and_groups','source':str((folder/'all_metrics.csv').resolve()),
             'note':'All classes, class macro, single/mixed and label counts, with effectiveN.'}],
        'limitations':PLAN['limitations']+[SCOPE], 'efficiency_experiment_included':False}
    record(folder/'paper_display.json',display)
    summary={'signature':signature,'context_signature':manifest['signature'],'evaluations':evaluations,
        'all_metrics':allrows,'paper_main_metrics':main_rows,'paired_comparisons':comparisons,
        'decisions':conclusion,'augmentation_audits':audits,'input_augmentation_audit':input_audit,
        'training_intervention_exposure':{name:histories[name] for name in MODELS[1:]},
        'figures':figures,'paper_display':display,'new_evaluations':new,'historical_references':replay,
        'scope':SCOPE,'limitations':PLAN['limitations']}
    record(folder/'summary.json',summary)
    for path,sha in binding.items():
        if file_hash(path)!=sha:raise ValueError('Reporting source changed during compilation')
    if any(file_hash(path)!=sha for path,sha in code.items()):raise ValueError('Reporting code changed during compilation')
    seal(folder,signature)
    return summary
