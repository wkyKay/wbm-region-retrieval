"""Render diagnostic figures from frozen JSON/CSV, without recomputing models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STATES = ('random', 'best_retrieval', 'best_nce')
NAMES = ('Initialization', 'Epoch 15', 'Epoch 100')
COLORS = ('#9aabb8', '#d48b31', '#247b9f')


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    root = Path(output)
    summary = json.loads((root/'summary.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140, 'savefig.dpi': 180})
    out = root/'figures'
    out.mkdir(exist_ok=True)

    def geometry(state, space, subset='valid_representative'):
        return summary['geometry'][f'{state}/{subset}/{space}']['class_balanced']['metrics']

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    x = np.arange(3)
    for row,space in enumerate(('h512','z128')):
        same = [geometry(s,space)['balanced_intra_mean_cosine'] for s in STATES]
        other = [geometry(s,space)['balanced_inter_mean_cosine'] for s in STATES]
        axes[row,0].plot(x,same,'o-',label='Same class, different wafer',color='#247b9f')
        axes[row,0].plot(x,other,'s--',label='Different class (balanced)',color='#c95d55')
        axes[row,0].set(title=f'{space}: raw cosine similarity', xticks=x, xticklabels=NAMES,
                        ylabel='Mean cosine (no clipping)')
        axes[row,0].legend(fontsize=8)
        vals = [geometry(s,space)['balanced_cosine_margin'] for s in STATES]
        axes[row,1].bar(x,vals,color=COLORS,width=.6)
        axes[row,1].set(title=f'{space}: same minus different class',xticks=x,xticklabels=NAMES,
                        ylabel='Balanced cosine margin')
        axes[row,1].axhline(0,color='#999',linewidth=.7)
        for i,v in enumerate(vals): axes[row,1].annotate(f'{v:.4f}',(i,v),ha='center',xytext=(0,4),textcoords='offset points')
    fig.suptitle('Single-label valid wafers | largest region per map | class-balanced\nAbsolute spread and class separation are different questions',fontsize=13)
    fig.savefig(out/'class_geometry.png'); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,space in zip(axes,('h512','z128')):
        for i,state in enumerate(STATES):
            vals=[geometry(state,space)[k]*100 for k in ('hit_at_1','precision_at_5','precision_at_10')]
            ax.bar(np.arange(3)+(i-1)*.24,vals,width=.23,color=COLORS[i],label=NAMES[i])
        ax.set(title=space,xticks=np.arange(3),xticklabels=('1-NN accuracy','P@5','P@10'),ylabel='Class-balanced %',ylim=(0,105))
        ax.legend(fontsize=8)
    fig.suptitle('Diagnostic single-label gallery | not the official full-gallery retrieval protocol',fontsize=12)
    fig.savefig(out/'single_label_neighbors.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    view_names=('rot90','context_drop','target_only')
    for ax,space in zip(axes,('h512','z128')):
        for i,state in enumerate(STATES):
            vals=[summary['context'][state][f'{v}/{space}/mixed']['common_context_cohort']['mean']['cosine_distance'] for v in view_names]
            ax.bar(np.arange(3)+(i-1)*.24,vals,width=.23,color=COLORS[i],label=NAMES[i])
        ax.set(title=space,xticks=np.arange(3),xticklabels=('90-degree rotation','Drop 50% context','Remove all context'),
               ylabel='1 - cosine to original')
        ax.legend(fontsize=8)
    fig.suptitle('Mixed-label valid maps | identical regions with active context dropout\nTargets and wafer support preserved; context can contain parts of the same defect',fontsize=11)
    fig.savefig(out/'context_sensitivity.png');plt.close(fig)

    # Illustrative inputs are selected by frozen identity, not by observed effect.
    with np.load(root/'context_examples.npz') as samples:
        names=('original','rot90','context_drop','target_only')
        fig,axes=plt.subplots(3,4,figsize=(9,7),constrained_layout=True)
        for row in range(min(3,len(samples['indices']))):
            for col,name in enumerate(names):
                patch=samples[name][row]
                rgb=np.ones((*patch.shape[-2:],3))
                rgb[patch[2]==1]=(.85,.88,.9)
                rgb[patch[0]==1]=(.25,.35,.4)
                rgb[patch[1]==1]=(.9,.3,.18)
                axes[row,col].imshow(rgb,interpolation='nearest')
                axes[row,col].set_title(name if row==0 else '')
                axes[row,col].axis('off')
        fig.suptitle('Fixed-ID examples | target in orange, other defects in dark gray',fontsize=12)
        fig.savefig(out/'context_examples.png');plt.close(fig)

    lines=['F 区域表示诊断（train/valid；不是正式检索成绩）',summary['identity']['run_id'],'',
           '代表区域为每张单类晶圆的最大原始cc8区域；单类图标签是区域语义的代理，不是像素真值。',
           '各类别等权；候选排除同晶圆所有区域；cosine没有裁剪。','']
    for space in ('h512','z128'):
        lines.append(space+'，valid代表区域：')
        for state in STATES:
            m=geometry(state,space)
            lines.append(f"{state}: 同类={m['balanced_intra_mean_cosine']:.6f}, 异类={m['balanced_inter_mean_cosine']:.6f}, "
                         f"间隔={m['balanced_cosine_margin']:.6f}, 1NN={m['hit_at_1']*100:.4f}%, P10={m['precision_at_10']*100:.4f}%")
        change=summary['comparisons'][f'best_retrieval_to_best_nce/valid_representative/{space}']['effects']
        for key in ('balanced_cosine_margin','hit_at_1','precision_at_10'):
            e=change[key]
            lines.append(f"epoch100-epoch15 {key}: {e['difference']:.6f}, paired95%CI [{e['ci_low']:.6f},{e['ci_high']:.6f}]")
        lines.append('')
    s=summary['negative_sampling']['class_balanced']
    lines += [f"每次5000负样本，同类跨晶圆负例期望数量（类均衡）: {s['expected_sameclass_crossmap_count']:.4f}",
              f"至少抽中1个的概率（类均衡）: {s['probability_sameclass_crossmap_at_least_one']:.8f}",'',
              '梯度仅是冻结checkpoint/BN与memory下的z空间瞬时压力；并非历史真实优化轨迹或因果训练实验。',
              '绝对同类相似度下降必须与异类、margin、近邻及各类别一起解释；不能单独判定表示变差。',
              '第15轮是此前按valid检索选出的辅助checkpoint；本轮不更改主方法或重新选checkpoint。',
              '完整数值见summary.json、paired_comparisons.json及各state目录。']
    (root/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    render(sys.argv[1])
