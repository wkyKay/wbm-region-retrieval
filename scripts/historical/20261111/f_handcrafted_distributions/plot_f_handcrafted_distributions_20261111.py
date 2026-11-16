"""Publication-ready histograms from exhaustive fixed-test statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
root=Path(a.run)
stats=json.loads((root/'statistics.json').read_text())
fonts=[f for f in font_manager.fontManager.ttflist if 'Noto Sans CJK' in f.name]
assert fonts, 'A CJK font is required'
plt.rcParams.update({'font.family':fonts[0].name,'axes.unicode_minus':False,'font.size':11,
    'axes.titlesize':14,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
    'savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
with np.load(root/'histograms.npz') as z:
    hist={k:z[k] for k in z.files}
edges=np.linspace(0,1,101)
colors={'F':'#2864B4','Handcrafted':'#D47C26'}
names={'F':'F（自监督 h512）','Handcrafted':'Handcrafted A（Zernike）'}
def percentages(key):
    counts=hist[key].reshape(100,10).sum(1)
    return 100*counts/counts.sum()
def base(ax):
    ax.set_xlim(0,1)
    ax.set_xticks(np.linspace(0,1,6))
    ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.set_xlabel('分数')
    ax.set_ylabel('配对占比（每箱宽度 0.01）')
    ax.grid(axis='y',color='#D9DEE5',alpha=.65,linewidth=.7)
    ax.set_axisbelow(True)
def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(root/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13.8,5.7))
fig.subplots_adjust(left=.065,right=.985,top=.76,bottom=.21,wspace=.24)
for ax,domain,title,count in zip(axes,('region','whole_map'),('区域对相似度','聚合后的整图匹配分数'),
                              (813105740,51818402)):
    base(ax)
    ymax=0
    for method in ('F','Handcrafted'):
        key=f'region_{method}' if domain=='region' else f'map_{method}_all'
        values=percentages(key);ymax=max(ymax,float(values.max()))
        ax.stairs(values,edges,fill=True,color=colors[method],alpha=.20,linewidth=0)
        ax.stairs(values,edges,color=colors[method],linewidth=1.6)
    ax.axvline(.5,color='#8B939F',linestyle='--',linewidth=1)
    ax.set_ylim(0,ymax*1.2)
    ax.set_title(f'{title}\n全部 {count:,} 个有向配对',pad=12)
    means={m:stats[domain][m]['mean'] if domain=='region' else stats[domain][m]['all']['mean'] for m in colors}
    ax.text(.97,.96,f'均值  F：{means["F"]:.4f}\nHandcrafted：{means["Handcrafted"]:.4f}',
            transform=ax.transAxes,ha='right',va='top',fontsize=10,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':4})
fig.suptitle('F 与 Handcrafted：完整 test 分数分布',fontsize=19,y=.985)
fig.legend([Patch(facecolor=colors[m],alpha=.7) for m in colors],[names[m] for m in colors],
           loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=12)
fig.text(.5,.085,'同一 test 集与区域；排除同一晶圆；阈值均为 0；整图使用相同的贪心匹配和 query 面积平方根加权。',
         ha='center',fontsize=10,color='#435062')
fig.text(.5,.035,'灰虚线仅标出 0.5 作为参考。两种方法的分数尺度不同，绝对高低不能直接解释为检索质量。',
         ha='center',fontsize=10,color='#435062')
save(fig,'score_histograms')

fig,axes=plt.subplots(2,2,figsize=(13.8,9.3))
fig.subplots_adjust(left=.07,right=.985,top=.87,bottom=.11,wspace=.23,hspace=.46)
for row,method in enumerate(('F','Handcrafted')):
    for col,(positive,negative,title,poslabel,neglabel) in enumerate((
        ('exact','not_exact','Exact：完整标签组合是否相同','相同组合','不同组合'),
        ('full_coverage','not_full_coverage','Coverage：是否完整覆盖 query 标签','完整覆盖','未完整覆盖'))):
        ax=axes[row,col];base(ax)
        ymax=0
        for group,label,color in ((positive,poslabel,'#247A64'),(negative,neglabel,'#B46574')):
            values=percentages(f'map_{method}_{group}');ymax=max(ymax,float(values.max()))
            mean=stats['whole_map'][method][group]['mean']
            ax.stairs(values,edges,fill=True,color=color,alpha=.17,linewidth=0)
            ax.stairs(values,edges,color=color,linewidth=1.6,label=f'{label}（均值 {mean:.3f}）')
        ax.set_title(names[method]+'\n'+title,pad=10)
        ax.set_ylim(0,ymax*1.25)
        ax.legend(loc='upper left' if method=='F' else 'upper right',frameon=True,framealpha=.93,fontsize=9)
fig.suptitle('整图分数：按两类评价的相关性分别观察',fontsize=18,y=.975)
fig.text(.5,.035,'每条分布在各自组内归一化为 100%；横轴一致。标签仅用于事后分组，不参与打分，也不是区域级真值。',
         ha='center',fontsize=10,color='#435062')
save(fig,'relevance_histograms')
print(json.dumps({'font':fonts[0].name,'saved':['score_histograms.png','score_histograms.pdf','score_histograms.svg',
    'relevance_histograms.png','relevance_histograms.pdf','relevance_histograms.svg']},ensure_ascii=False))
