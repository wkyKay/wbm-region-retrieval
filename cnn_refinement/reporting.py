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
COLORS={'baseline':'#4b5563','context_dropout':'#0072b2','filtered_negatives':'#d55e00'}
LABELS={'baseline':'Baseline','context_dropout':'Context dropout','filtered_negatives':'Filtered negatives'}
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


def context_statistics(rows_by_model,*,repetitions=2000,seed=2026):
    """Mixed-wafer h512 distance on one common actual-dropout cohort.

    Map means receive equal weight within each complete combination, then
    combinations receive equal weight, matching the diagnostic's primary mean.
    """
    indexed={}
    for name,rows in rows_by_model.items():
        selected=[r for r in rows if r['space']=='h512' and int(r['label_count'])>1]
        lookup={(r['view'],int(r['map_id']),int(r['region_id'])):r for r in selected}
        if len(lookup)!=len(selected):raise ValueError('Duplicate context region/view')
        indexed[name]=lookup
    base=indexed['baseline']
    common={(sid,rid) for (view,sid,rid),row in base.items()
            if view=='context_drop' and row['has_context'] and not row['is_noop']}
    if not common:raise ValueError('No non-noop mixed context cohort')
    output=[]
    for view in ('context_drop','target_only'):
        bymodel={}
        for name in MODELS:
            groups=defaultdict(list)
            for sid,rid in sorted(common):
                a=base[(view,sid,rid)];b=indexed[name][(view,sid,rid)]
                if any(a[k]!=b[k] for k in ('label_signature','token_index','is_noop','has_context',
                                           'valid_region_input','source_all_defect_changed_pixels')):
                    raise ValueError('Context model comparison changed its intervention cohort')
                if not b['valid_region_input']:raise ValueError('Invalid context input in primary report')
                groups[(a['label_signature'],sid)].append(float(b['cosine_distance']))
            bymodel[name]={key:float(np.mean(values)) for key,values in groups.items()}
        keys=sorted(bymodel['baseline']);labels=[key[0] for key in keys]
        reference=[bymodel['baseline'][key] for key in keys]
        for name in MODELS:
            candidate=[bymodel[name][key] for key in keys]
            means,draws=_paired_arrays(labels,reference,candidate,repetitions,seed,'class_macro')
            low,high=np.quantile(draws[:,1],[.025,.975])
            dlow,dhigh=np.quantile(draws[:,1]-draws[:,0],[.025,.975])
            valid=draws[:,0]>0
            reductions=1-draws[valid,1]/draws[valid,0]
            rlow,rhigh=np.quantile(reductions,[.025,.975]) if len(reductions) else (None,None)
            output.append({'model':name,'view':view,'space':'h512','group':'mixed_common_context',
                'regions':len(common),'maps':len(keys),'classes':len(set(labels)),
                'mean_distance':float(means[1]),'ci_low':float(low),'ci_high':float(high),
                'difference':float(means[1]-means[0]),'difference_ci_low':float(dlow),'difference_ci_high':float(dhigh),
                'relative_reduction':float(1-means[1]/means[0]) if means[0]>0 else None,
                'relative_reduction_ci_low':None if rlow is None else float(rlow),
                'relative_reduction_ci_high':None if rhigh is None else float(rhigh),
                'relative_reduction_effective_draws':int(valid.sum()),'repetitions':repetitions,
                'aggregation':'equal combination mean of equal wafer means; same actual-dropout regions',
                'scope':'Paired wafer clusters resampled within complete combination; fixed model and selected cohort.'})
    return output


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


def decisions(comparisons,context,diagnostics):
    rules=PLAN['decision_rules'];result={}
    lookup={(row['model'],row['stage'],row['style'],row['metric'],row['aggregation']):row for row in comparisons}
    contexts={(row['model'],row['view']):row for row in context}
    geometry=diagnostics['geometry']
    baseline=geometry['baseline/valid_representative/h512']
    for model in MODELS[1:]:
        exact=lookup[(model,'test','fusion','Precision@10','query_mean')]
        cross=lookup[(model,'cross_test','fusion','Coverage-nDCG@10','query_mean')]
        cnn=lookup[(model,'test','cnn_only','Precision@10','query_mean')]
        current=geometry[model+'/valid_representative/h512']
        nn={name:{'baseline':baseline['class_balanced']['metrics'][name],
                  'candidate':current['class_balanced']['metrics'][name]}
            for name in ('hit_at_1','precision_at_10')}
        for value in nn.values():value['difference']=value['candidate']-value['baseline']
        reductions={view:contexts[(model,view)]['relative_reduction'] for view in ('context_drop','target_only')}
        zero=bool(current['dispersion']['zero_normalized_variance'])
        criterion_exact=(exact['difference'] is not None and exact['difference']>=rules['fusion_exact_P10_absolute_improvement']
                         and exact['ci_low']>rules['paired_95CI_lower_above'])
        criterion_cross=cross['ci_low'] is not None and cross['ci_low']>rules['fusion_cross_nDCG10_noninferiority_lower']
        criterion_cnn=cnn['difference'] is not None and cnn['difference']>0
        context_threshold=all(value is not None and value>=rules['context_distance_relative_reduction'] for value in reductions.values())
        result[model]={'test_fusion_exact_P10':exact,'test_fusion_cross_nDCG10':cross,'test_cnn_only_exact_P10':cnn,
            'fusion_exact_P10_criterion_met':bool(criterion_exact),'fusion_cross_nDCG10_noninferiority_met':bool(criterion_cross),
            'cnn_only_P10_observed_improvement':bool(criterion_cnn),'cnn_only_P10_positive_CI':cnn['ci_low'] is not None and cnn['ci_low']>0,
            'main_retrieval_criteria_met':bool(criterion_exact and criterion_cross and criterion_cnn),
            'relative_context_reduction':reductions,'context_distance_criterion_met':context_threshold,
            'valid_singleton_class_balanced_NN':nn,'NN_observed_nondegradation':all(v['difference']>=0 for v in nn.values()),
            'baseline_dispersion':baseline['dispersion'],'candidate_dispersion':current['dispersion'],
            'zero_normalized_variance_detected':zero,
            'context_conditions_observed':bool(context_threshold and not zero and all(v['difference']>=0 for v in nn.values())),
            'interpretation':'Observed conditions for this fixed seed. No arbitrary variance/effective-rank cutoff is introduced; report the full values. CNN-only improvement uses positive observed exact P@10 delta, with its CI separately displayed.'}
    return result


def _save_figure(figure,folder,stem):
    paths=[]
    for suffix in ('png','pdf','svg'):
        path=folder/(stem+'.'+suffix)
        figure.savefig(path,dpi=240,bbox_inches='tight')
        paths.append(str(path.resolve()))
    return paths


def render_figures(folder,histories,context,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    result={}
    def intervals(ax,x,y,low,high,color,label):
        x,y,low,high=map(np.asarray,(x,y,low,high))
        # Draw the percentile interval itself, even in the unusual case where
        # a finite Monte Carlo percentile interval excludes the point estimate.
        ax.errorbar(x,(low+high)/2,yerr=(high-low)/2,fmt='none',capsize=4,color=color)
        ax.plot(x,y,'o',color=color,label=label)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for name,history in histories.items():
        for ax,metric in zip(axes,('train_loss','valid_nce')):
            ax.plot([r['epoch'] for r in history],[r[metric] for r in history],color=COLORS[name],label=LABELS[name],linewidth=1.5)
            ax.set_xlabel('Epoch');ax.set_ylabel('NCE loss');ax.grid(alpha=.15)
    axes[0].set_title('Training objectives (different interventions)')
    axes[1].set_title('Common original-view validation NCE')
    axes[0].legend(frameon=False,fontsize=8)
    fig.text(.5,-.03,'Loss curves describe optimization; lower training loss across objectives does not establish better retrieval.',ha='center',fontsize=9)
    fig.tight_layout();result['training_nce']=_save_figure(fig,folder,'training_nce');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    positions={'context_drop':0.,'target_only':1.}
    offsets={'baseline':-.23,'context_dropout':0.,'filtered_negatives':.23}
    for name in MODELS:
        rows=[r for r in context if r['model']==name]
        for i,metric in enumerate(('mean_distance','relative_reduction')):
            chosen=[r for r in rows if _finite(r[metric]) is not None and (i==0 or name!='baseline')]
            if not chosen:continue
            x=[positions[r['view']]+offsets[name] for r in chosen]
            y=np.asarray([r[metric] for r in chosen])
            lo=np.asarray([r['ci_low' if i==0 else 'relative_reduction_ci_low'] for r in chosen])
            hi=np.asarray([r['ci_high' if i==0 else 'relative_reduction_ci_high'] for r in chosen])
            intervals(axes[i],x,y,lo,hi,COLORS[name],LABELS[name])
    for ax in axes:
        ax.set_xticks([0,1],['Drop 50% context','Remove all context']);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('h512 cosine distance');axes[1].set_ylabel('Relative reduction vs baseline')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].axhline(.30,color='#999',linestyle='--',linewidth=1)
    axes[0].set_title('Common mixed-wafer cohort');axes[1].set_title('Paired wafer-cluster 95% intervals')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,fontsize=9)
    first=context[0]
    fig.suptitle(f'Common non-noop mixed cohort: {first["regions"]} regions in {first["maps"]} wafers',y=1.07,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.88));result['context_common_mixed']=_save_figure(fig,folder,'context_common_mixed');plt.close(fig)
    for split in ('valid','test'):
        fig,axes=plt.subplots(1,3,figsize=(13,3.9))
        panels=((split,'Precision@10','Ordinary exact P@10'),('cross_'+split,'Coverage-nDCG@10','Cross-combination nDCG@10'),
                ('cross_'+split,'FullCoveragePrecision@10','Cross-combination full-coverage P@10'))
        for ax,(stage,metric,title) in zip(axes,panels):
            for j,name in enumerate(MODELS[1:]):
                rows=[r for r in comparisons if r['model']==name and r['stage']==stage and r['metric']==metric and r['aggregation']=='query_mean']
                rows=sorted(rows,key=lambda r:STYLES.index(r['style']))
                y=np.asarray([r['difference'] for r in rows])*100
                lo=np.asarray([r['ci_low'] for r in rows])*100;hi=np.asarray([r['ci_high'] for r in rows])*100
                x=np.arange(2)+(j-.5)*.15
                intervals(ax,x,y,lo,hi,COLORS[name],LABELS[name])
            controls=[r for r in comparisons if r['model']=='context_dropout' and r['stage']==stage and r['metric']==metric and r['aggregation']=='query_mean']
            controls=sorted(controls,key=lambda r:STYLES.index(r['style']))
            ax.text(.5,-.22,'Baseline CNN / fusion: '+' / '.join(f'{r["reference"]:.3f}' for r in controls)
                    +'\nPaired N: '+' / '.join(str(r['paired_effective_n']) for r in controls),
                    transform=ax.transAxes,ha='center',fontsize=7)
            ax.axhline(0,color='#777',linewidth=1);ax.grid(axis='y',alpha=.15)
            ax.set_xticks([0,1],['CNN only','Fixed fusion']);ax.set_title(title,fontsize=10)
            ax.set_ylabel('Change vs baseline (percentage points)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,fontsize=9)
        fig.suptitle(split.capitalize()+': paired query 95% intervals, fixed models and gallery',y=1.07,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.88));result['retrieval_'+split]=_save_figure(fig,folder,'retrieval_'+split);plt.close(fig)
    return result


def build_report(output,bootstrap=2000):
    """Require all 24 evaluations and completed diagnostics, then seal a report."""
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
                    for family,metric,column in (('exact','Precision@10','exact_P10'),('exact','mAP@10','exact_mAP10'),
                        ('coverage','Coverage-nDCG@10','coverage_nDCG10'),('coverage','FullCoveragePrecision@10','full_coverage_P10'),
                        ('coverage','FullCoverageRecall@10','full_coverage_R10'),('coverage','FullCoverage_mAP@10','full_coverage_mAP10')):
                        document=documents.get(family,{})
                        compact[column]=document.get('metrics',{}).get(metric)
                        compact[column+'_effective_n']=document.get('metric_query_counts',{}).get(metric)
                    main_rows.append(compact)
    if (len(evaluations),new,replay)!=(24,20,4):raise ValueError('Expected 24 evaluations: 20 new and four original baseline-fusion references')
    histories={}
    baseline_history=Path(manifest['baseline']['checkpoint']).parent/'history.json'
    for name in MODELS:
        path=baseline_history if name=='baseline' else output/'training'/name/'history.json'
        if name!='baseline':_sealed_files(path.parent,['history.json','training_summary.json'],binding)
        history=read_json(_bind(path,binding))
        if [r['epoch'] for r in history]!=list(range(1,101)) or any(_finite(r[m]) is None for r in history for m in ('train_loss','valid_nce')):
            raise ValueError('Training history must contain all 100 completed epochs')
        histories[name]=history
    diagnostic=output/'diagnostics'
    _sealed_files(diagnostic,['summary.json']+[m+'/context_rows.json' for m in MODELS],binding)
    diagnostics=read_json(diagnostic/'summary.json')
    context_rows={m:read_json(diagnostic/m/'context_rows.json') for m in MODELS}
    _sealed_files(output/'negative_diagnostics',['summary.json'],binding)
    negative=read_json(output/'negative_diagnostics/summary.json')
    code={str(Path(__file__).with_name(name).resolve()):file_hash(Path(__file__).with_name(name))
          for name in ('reporting.py','state.py','register.py')}
    science={'context_signature':manifest['signature'],'source_binding':dict(sorted(binding.items())),
             'code':code,'bootstrap':bootstrap,'seed':2026,'decision_rules':PLAN['decision_rules'],
             'new_evaluations':new,'historical_references':replay,'scope':SCOPE}
    signature=digest(science)
    if verified(folder,signature):return read_json(folder/'summary.json')
    contexts=context_statistics(context_rows,repetitions=bootstrap)
    comparisons=[]
    for model in MODELS[1:]:
        for stage in STAGES:
            targets=[('coverage','Coverage-nDCG@10'),('coverage','FullCoveragePrecision@10')]
            if not stage.startswith('cross_'):targets.insert(0,('exact','Precision@10'))
            for style in STYLES:
                for family,metric in targets:
                    for aggregate in ('query_mean','class_macro'):
                        comparison=paired_query_bootstrap(perquery[('baseline',stage,style,family)],
                            perquery[(model,stage,style,family)],metric,repetitions=bootstrap,
                            aggregation=aggregate,expected_classes=37)
                        comparisons.append({'model':model,'stage':stage,'style':style,'family':family,**comparison})
    conclusion=decisions(comparisons,contexts,diagnostics)
    folder.mkdir(parents=True,exist_ok=True)
    record(folder/'manifest.json',{**science,'signature':signature})
    write_csv(folder/'all_metrics.csv',allrows,list(allrows[0]))
    write_csv(folder/'paper_main_metrics.csv',main_rows,list(main_rows[0]))
    write_csv(folder/'paired_retrieval_differences.csv',comparisons,list(comparisons[0]))
    write_csv(folder/'context_common_mixed.csv',contexts,list(contexts[0]))
    historyrows=[{'model':name,**row} for name,history in histories.items() for row in history]
    fields=list(dict.fromkeys(key for row in historyrows for key in row))
    write_csv(folder/'training_history.csv',historyrows,fields)
    negative_rows=_scalar_table(negative)
    write_csv(folder/'negative_diagnostics_summary.csv',negative_rows,['metric','value'])
    figures=render_figures(folder/'figures',histories,contexts,comparisons)
    display={'main_text':[
        {'figure':'retrieval_test','files':figures['retrieval_test'],'metrics':['ordinary exact Precision@10','cross Coverage-nDCG@10','cross FullCoveragePrecision@10'],
         'required_annotation':'baseline absolute values, model differences, paired 95% CI and metric-specific effective N; CNN-only and fusion separately'},
        {'figure':'context_common_mixed','files':figures['context_common_mixed'],
         'metrics':['h512 cosine distance for drop50 and dropall','relative distance reduction with paired wafer-cluster 95% CI'],
         'required_annotation':'same actual-dropout mixed-wafer cohort; wafer/region counts; this is sensitivity, not assumed semantic invariance'},
        {'table':'comparison_and_decisions','source':str((folder/'paper_main_metrics.csv').resolve()),
         'decision_details':str((folder/'summary.json').resolve()),
         'metrics':['fusion exact P@10 point gain and CI','cross nDCG@10 noninferiority CI','CNN-only exact P@10 gain and CI',
                    'singleton 1NN and P@10','normalized variance and effective rank','negative replacement exposure and label-proxy audit']}],
        'appendix':[
            {'figure':'training_nce','files':figures['training_nce'],'note':'Training objectives differ; curves do not rank methods by loss.'},
            {'figure':'retrieval_valid','files':figures['retrieval_valid'],'note':'Validation observations were not used to reselect epochs or methods.'},
            {'table':'all_seven_K_and_groups','source':str((folder/'all_metrics.csv').resolve()),
             'note':'All classes, class macro, single/mixed and label-count groups; preserve metric-specific valid N.'}],
        'limitations':PLAN['limitations']+[SCOPE], 'efficiency_experiment_included':False}
    record(folder/'paper_display.json',display)
    summary={'signature':signature,'context_signature':manifest['signature'],'evaluations':evaluations,
        'all_metrics':allrows,'paper_main_metrics':main_rows,'paired_comparisons':comparisons,'context_common_mixed':contexts,
        'decisions':conclusion,'diagnostics':diagnostics,'negative_diagnostics':negative,
        'training_intervention_exposure':{name:histories[name] for name in MODELS[1:]},
        'figures':figures,'paper_display':display,'new_evaluations':new,'historical_references':replay,
        'scope':SCOPE,'limitations':PLAN['limitations']}
    record(folder/'summary.json',summary)
    for path,sha in binding.items():
        if file_hash(path)!=sha:raise ValueError('Reporting source changed during compilation')
    if any(file_hash(path)!=sha for path,sha in code.items()):raise ValueError('Reporting code changed during compilation')
    seal(folder,signature)
    return summary
