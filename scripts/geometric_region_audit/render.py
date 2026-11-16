"""Static image sheets and a self-contained pixel-inspection viewer."""
from __future__ import annotations

import colorsys
import html
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKGROUND = (255, 255, 255)
NORMAL = (231, 235, 239)
UNASSIGNED = (35, 40, 48)


def palette(count):
    fixed = ["008F86", "E18317", "7460C5", "D64B75", "287DC2", "7C9F24", "AA5B26", "5B76B3"]
    out = [list(UNASSIGNED)]
    for i in range(count):
        if i < len(fixed):
            out.append([int(fixed[i][j:j+2], 16) for j in (0, 2, 4)])
        else:
            hue = ((i - len(fixed)) * .618033988749895 + .04) % 1
            sat = (.70, .88, .57)[i % 3]
            val = (.82, .96, .70)[(i // 3) % 3]
            rgb = [int(round(v * 255)) for v in colorsys.hsv_to_rgb(hue, sat, val)]
            while rgb in out:
                rgb[2] = (rgb[2] + 1) % 256
            out.append(rgb)
    return out


def rgb_map(raw, labels=None):
    rgb = np.full((*raw.shape, 3), BACKGROUND, np.uint8)
    rgb[raw == 1] = NORMAL
    rgb[raw == 2] = UNASSIGNED
    if labels is not None:
        colors = np.asarray(palette(int(labels.max())), np.uint8)
        foreground = labels > 0
        rgb[foreground] = colors[labels[foreground]]
    return rgb


def panel(ax, raw, labels, title, number_regions=False):
    ax.imshow(rgb_map(raw, labels), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cdd5de")
        spine.set_linewidth(.6)
    ax.set_title(title, fontsize=10, color="#172b42", pad=7)
    total = int((raw == 2).sum())
    if labels is None:
        footer = f"{total} original defect pixels"
    else:
        n, kept = int(labels.max()), int((labels > 0).sum())
        pct = f"{100*kept/total:.1f}%" if total else "n/a"
        footer = f"{n} regions | kept {pct}\ndropped {total-kept} px"
        if number_regions:
            # Labels are placed on an actual region pixel, not a centroid in a hole.
            for rid in range(1, n + 1):
                points = np.argwhere(labels == rid)
                if len(points) < 20:
                    continue
                point = points[np.argmin(np.square(points - points.mean(0)).sum(1))]
                ax.text(point[1], point[0], str(rid), ha="center", va="center", fontsize=6,
                        color="white", bbox=dict(facecolor="#172b42", edgecolor="none", alpha=.8, pad=.7))
    ax.set_xlabel(footer, fontsize=8, color="#43536a", labelpad=6)


def write_sheets(output, samples, maps, results, methods):
    paths = []
    for mode, mode_name in (("native", "All native proposals"), ("f_budget", "F budget: area >= 5, top 4")):
        folder = output / "images" / mode
        folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(samples), 5):
            stop = min(start + 5, len(samples))
            height = (stop-start)*3.4+1.2
            fig, axes = plt.subplots(stop-start, len(methods)+1, figsize=(13.6, height), squeeze=False)
            for row, idx in enumerate(range(start, stop)):
                sample = samples[idx]
                class_title = textwrap.fill(f"{idx+1:02d}. {sample['class_name']}", width=28, break_long_words=False)
                title = f"{class_title}\nsource ID {sample['sample_id']}"
                panel(axes[row, 0], maps[idx], None, title)
                for col, method in enumerate(methods, 1):
                    panel(axes[row, col], maps[idx], results[mode][method['id']][idx], method['title'], True)
            fig.suptitle(f"MixedWM38 geometry audit | {mode_name} | {start+1}-{stop} / 38", fontsize=15, x=.03, ha="left", y=.99)
            fig.text(.03, .009, "Color = region ID (not defect class). Dark pixels = unassigned defects. Light gray = passing die.\nT-Linkage: line-model port. Multi-X: line/circle adaptation. See method notes for reproduction boundaries.", fontsize=8, color="#43536a")
            fig.subplots_adjust(left=.025, right=.995, top=1-1.12/height, bottom=.78/height, wspace=.16, hspace=.66)
            path = folder / f"sheet_{start//5+1:02d}.png"
            fig.savefig(path, dpi=150, facecolor="white")
            plt.close(fig)
            paths.append(str(path.relative_to(output)))
        # Every class also gets a full-resolution, one-row image.
        for idx, sample in enumerate(samples):
            fig, axes = plt.subplots(1, len(methods)+1, figsize=(13.6, 4.3))
            panel(axes[0], maps[idx], None, "Original")
            for col, method in enumerate(methods, 1):
                panel(axes[col], maps[idx], results[mode][method['id']][idx], method['title'], True)
            fig.suptitle(f"{idx+1:02d}. {sample['class_name']} | source ID {sample['sample_id']} | {mode_name}", fontsize=14, x=.025, ha="left")
            fig.text(.025, .02, "Colors identify regions within each panel; they are not semantic class labels. Dark pixels are excluded original defects. Line/circle models are geometric priors.", fontsize=9)
            fig.subplots_adjust(left=.025, right=.995, top=.79, bottom=.23, wspace=.16)
            fig.savefig(folder / f"class_{idx+1:02d}_id_{sample['sample_id']}.png", dpi=150, facecolor="white")
            plt.close(fig)
    return paths


def write_viewer(output, samples, maps, results, methods, stats, sheets):
    maximum = max(int(a.max()) for mode in results.values() for a in mode.values())
    payload = dict(samples=samples, maps=maps.tolist(), methods=methods, stats=stats,
                   results={mode: {k: v.tolist() for k, v in values.items()} for mode, values in results.items()},
                   palette=palette(maximum), sheets=sheets)
    template = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MixedWM38 · T-Linkage / Multi-X 区域审阅</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#182d43;font:15px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1600px;margin:auto;padding:26px}h1{margin:0;font-size:28px}h2{font-size:19px}p{margin:9px 0}.sub{color:#566779}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:20px 0}button,select{font:inherit;border:1px solid #c7d2dd;border-radius:7px;background:white;color:inherit;padding:8px 12px;cursor:pointer}select#sample{max-width:500px}button:hover{background:#eaf1f6}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}article{background:white;border:1px solid #d7e0e8;border-radius:10px;padding:14px}article h3{font-size:16px;margin:0 0 5px}.tag{font-size:12px;color:#657587;min-height:36px}.map{width:100%;aspect-ratio:1;image-rendering:pixelated;display:block;border:1px solid #d7e0e8;margin-top:6px;cursor:crosshair}.count{font-size:14px;margin-top:7px}.info{min-height:28px;font-size:13px;color:#536679}.legend{background:#eaf0f5;padding:12px 16px;border-radius:8px}.legend span{display:inline-block;margin-right:18px}.swatch{display:inline-block;width:11px;height:11px;margin-right:6px;border:1px solid #b6c3cf}.links a,a{color:#006f83}details{margin-top:20px;background:white;border:1px solid #d7e0e8;border-radius:8px;padding:14px}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #e0e6ed;padding:10px;vertical-align:top}#title{margin-bottom:4px}.badges{display:flex;gap:7px;flex-wrap:wrap;margin-top:15px}.badge{font-size:12px;background:#e4edf2;padding:4px 9px;border-radius:20px}.sheets a{display:inline-block;margin-right:14px;margin-bottom:9px}#thumbs{display:flex;flex-wrap:wrap;gap:5px;margin:15px 0}.thumb{padding:3px;width:52px}.thumb canvas{display:block;width:44px;height:44px;image-rendering:pixelated}.thumb.active{border:2px solid #008f86}footer{color:#657587;font-size:13px;margin:24px 0} .note{color:#6a4a19;background:#fff5e0;padding:12px 16px;border-radius:8px}
</style><main>
<h1>MixedWM38 · T-Linkage / Multi-X 区域审阅</h1>
<p class="sub">沿用上一轮 38 类固定样例 · seed=2026 · 仅 train · 原始 52×52 网格 · 无类别标签训练</p>
<div class="badges"><span class="badge">类别仅用于抽样与标题</span><span class="badge">每种方法使用同一张图</span><span class="badge">全部输出都是真实缺陷像素子集</span></div>
<p class="note">并排展示原图、8 连通、T-Linkage 多直线、Multi-X 直线＋圆适配结果。所有类别使用同一组参数；颜色表示算法分组，不表示缺陷语义类别。几何模型可能把块状缺陷拆成许多线／圆，原生视图保留这些结果供判断。交点只有一个区域编号。本次是方法移植与晶圆适配验证，具体复现边界见下方方法说明。</p>
<div class="toolbar"><button id="prev">← 上一类</button><select id="sample"></select><button id="next">下一类 →</button><select id="mode"><option value="native">全部原生候选（不额外套 F 过滤）</option><option value="f_budget">F 预算：面积 ≥ 5，最多 4 区域</option></select></div>
<div class="legend"><span><i class="swatch" style="background:#008f86"></i>彩色：不同区域</span><span><i class="swatch" style="background:#232830"></i>深灰：未分配／过滤的缺陷点</span><span><i class="swatch" style="background:#e7ebef"></i>浅灰：正常 die</span><span><i class="swatch" style="background:white"></i>白色：0/3（不参与提取）</span><br>颜色按区域面积降序编号；跨方法同色不保证对应同一模式。鼠标移到像素可查区域编号，点击可单独突出该区域，再点恢复。</div>
<div id="thumbs"></div><h2 id="title"></h2><p id="sampleinfo" class="sub"></p><div class="cards" id="cards"></div>
<p class="links" id="downloads"></p>
<details><summary>方法、参数与复现边界</summary><table><thead><tr><th>方法</th><th>状态</th><th>说明</th></tr></thead><tbody id="methodrows"></tbody></table><p>所有算法仅接收原始 raw==2 像素坐标，不输入类别标签、真实模式数量或基于类别的预处理。输出都是原图真实缺陷像素子集；没有绘制拟合曲线经过的正常像素，没有添加圆环内部的正常像素。区域编号在各面板内按面积降序排列。</p><p>论文：<a href="https://fusiello.github.io/papers/cvpr14.pdf">T-Linkage (CVPR 2014)</a>；<a href="https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Daniel_Barath_Multi-Class_Model_Fitting_ECCV_2018_paper.pdf">Multi-X (ECCV 2018)</a>。实现说明：<a href="TLINKAGE_NOTES.md">T-Linkage</a> · <a href="MULTIX_NOTES.md">Multi-X</a>。</p></details>
<details><summary>下载完整对照图与像素数据</summary><p><a href="synthetic_controls.png">合成相交线／线圆检查</a> · <a href="preview_native.png">固定混合类示例预览</a></p><h3>全部原生候选</h3><div class="sheets" id="native-sheets"></div><h3>F 预算过滤后</h3><div class="sheets" id="budget-sheets"></div><p><a href="regions_native.npz">原生区域像素 NPZ</a> · <a href="regions_f_budget.npz">F 预算区域像素 NPZ</a> · <a href="selected_samples.npz">38 张原图及标签</a> · <a href="sample_manifest.csv">样本 ID 清单</a> · <a href="statistics.csv">区域数量与保留像素统计</a> · <a href="validation.json">核查结果</a> · <a href="README.md">完整说明</a></p><p>NPZ 每个方法键为 [38,52,52] 整数区域编号：0表示未分配，1…K表示具体区域。第 i 个区域的二值 mask 可用 label_map==i 得到。未分配缺陷可用 (raw==2)&(label_map==0) 得到。</p></details>
<footer>每类仅一张用于肉眼检查，不代表分割准确率或检索性能。仅使用 train，不检查 test，不修改冻结划分、F 定义或旧产物。</footer>
</main><script>
const D=__DATA__;
const sample=document.querySelector('#sample'), mode=document.querySelector('#mode');
let index=0;
function color(raw,label){if(label>0)return D.palette[label];return raw===2?[35,40,48]:raw===1?[231,235,239]:[255,255,255]}
function paint(canvas,raw,labels,selected=0){canvas.width=raw[0].length;canvas.height=raw.length;let ctx=canvas.getContext('2d'),im=ctx.createImageData(canvas.width,canvas.height);for(let r=0;r<raw.length;r++)for(let c=0;c<raw[0].length;c++){let id=labels?labels[r][c]:0, rgb=color(raw[r][c],id);if(selected&&id!==selected&&raw[r][c]===2)rgb=[211,216,222];let p=(r*canvas.width+c)*4;im.data.set([...rgb,255],p)}ctx.putImageData(im,0,0)}
D.samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${String(i+1).padStart(2,'0')} · ${s.class_name} · ID ${s.sample_id}`;sample.append(o);const b=document.createElement('button');b.className='thumb';b.title=`${s.class_name} · ID ${s.sample_id}`;const c=document.createElement('canvas');b.append(c);paint(c,D.maps[i],null);b.onclick=()=>{index=i;show()};document.querySelector('#thumbs').append(b)});
function show(){sample.value=index;document.querySelectorAll('.thumb').forEach((e,i)=>e.classList.toggle('active',i===index));const s=D.samples[index],raw=D.maps[index],active=mode.value;document.querySelector('#title').textContent=`${index+1} / 38 · ${s.class_name}`;document.querySelector('#sampleinfo').textContent=`原始 NPZ 行号 ${s.sample_id} · ${s.split} · 原始缺陷 ${s.defect_pixels} 像素 · ${active==='native'?'算法原生候选':'统一 F 预算（8 连通列逐像素等于当前 F）'}`;const cards=document.querySelector('#cards');cards.replaceChildren();[{id:'original',name:'原始晶圆图',status:'原图，不滤波'},...D.methods].forEach(m=>{const article=document.createElement('article');const title=document.createElement('h3');title.textContent=m.name;article.append(title);let tag=document.createElement('div');tag.className='tag';tag.textContent=m.status;article.append(tag);let canvas=document.createElement('canvas');canvas.className='map';article.append(canvas);const labels=m.id==='original'?null:D.results[active][m.id][index];paint(canvas,raw,labels);const sizes={};if(labels)labels.flat().forEach(v=>{if(v>0)sizes[v]=(sizes[v]||0)+1});let kept=Object.values(sizes).reduce((a,b)=>a+b,0);const count=document.createElement('div');count.className='count';count.textContent=labels?`${Object.keys(sizes).length} 区域 · 保留 ${kept}/${s.defect_pixels} 像素${s.defect_pixels?' ('+(100*kept/s.defect_pixels).toFixed(1)+'%)':''}`:`${s.defect_pixels} 个缺陷像素`;article.append(count);let info=document.createElement('div');info.className='info';info.textContent='悬停像素查看坐标';article.append(info);let selected=0;function at(e){const b=canvas.getBoundingClientRect();return [Math.min(raw.length-1,Math.floor((e.clientY-b.top)*raw.length/b.height)),Math.min(raw[0].length-1,Math.floor((e.clientX-b.left)*raw[0].length/b.width))]};canvas.onmousemove=e=>{const [r,c]=at(e),id=labels?labels[r][c]:0;info.textContent=`(${r}, ${c}) · 原值 ${raw[r][c]} · `+(id?`区域 ${id} / ${sizes[id]} 像素`:raw[r][c]===2?(labels?'未分配缺陷':'原始缺陷'):'非缺陷')};canvas.onclick=e=>{if(!labels)return;const [r,c]=at(e),id=labels[r][c];selected=selected===id?0:id;paint(canvas,raw,labels,selected)};cards.append(article)});const link=document.createElement('a');link.href=`images/${active}/class_${String(index+1).padStart(2,'0')}_id_${s.sample_id}.png`;link.textContent='打开本类全部方法高清并排图';link.target='_blank';document.querySelector('#downloads').replaceChildren(link)}
sample.onchange=()=>{index=Number(sample.value);show()};mode.onchange=show;document.querySelector('#prev').onclick=()=>{index=(index+37)%38;show()};document.querySelector('#next').onclick=()=>{index=(index+1)%38;show()};document.addEventListener('keydown',e=>{if(['SELECT','INPUT'].includes(e.target.tagName))return;if(e.key==='ArrowRight'){index=(index+1)%38;show()}if(e.key==='ArrowLeft'){index=(index+37)%38;show()}});
D.methods.forEach(m=>{const tr=document.createElement('tr');[m.name,m.status,m.detail].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});document.querySelector('#methodrows').append(tr)});D.sheets.forEach(p=>{const a=document.createElement('a');a.href=p;a.target='_blank';a.textContent=p.split('/').at(-1);document.querySelector(p.includes('/native/')?'#native-sheets':'#budget-sheets').append(a)});show();
</script></html>'''
    text = template.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    (output / "index.html").write_text(text, encoding="utf-8")
