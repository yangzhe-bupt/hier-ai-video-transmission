"""用已有真实模型重建生成本地画质验收页；不修改图片、不训练。"""
import csv
import json
from pathlib import Path
import random
import shutil
import cv2

project=Path(__file__).resolve().parents[2]
source=project/'outputs/20261120_143206_105127'
out=project/'outputs/psnr_acceptance_20261120';out.mkdir(parents=True,exist_ok=True)
with (source/'metrics.csv').open(encoding='utf-8') as f:
    rows=[r for r in csv.DictReader(f) if r['eval_psnr_db'] and int(r['step'])>=500]
selected={int(min(rows,key=lambda r:abs(float(r['eval_psnr_db'])-target))['step']) for target in (24,26,28,30,32,34)}
chosen=[r for r in rows if int(r['step']) in selected]
random.Random(91).shuffle(chosen)
items=[]
for i,r in enumerate(chosen):
    step=int(r['step']);name=f'version_{chr(65+i)}.png'
    original=source/f'reconstruction_{step:04d}_valid.png'
    shutil.copy2(original,out/name)
    h,w=cv2.imread(str(original)).shape[:2]
    items.append(dict(label=chr(65+i),file=name,step=step,psnr_db=float(r['eval_psnr_db']),
        mse=float(r['eval_mse']),width=w,height=h,source=str(original)))
shutil.copy2(source/'metrics.csv',out/'source_metrics.csv')
(out/'sources.json').write_text(json.dumps(dict(source=str(source),frames=[2273,2274,2275,2276],
    note='original saved montage PNGs, CSS displays top/reference and bottom/reconstruction thirds; no synthetic blur',items=items),ensure_ascii=False,indent=2),encoding='utf-8')
template='''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>固定GOP画质验收</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1250px;margin:24px auto;padding:0 18px;background:#f4f6f8;color:#192633}
h1{font-size:25px}p{line-height:1.7}.panel{background:white;border:1px solid #d4dce4;padding:16px;margin:18px 0;border-radius:10px}
.crop{position:relative;overflow:hidden;width:100%;background:white}.crop img{position:absolute;width:100%;left:0}.reference img{top:0}.reconstruction img{bottom:0}
label{display:inline-block;margin:10px 22px 10px 0}button{padding:10px 18px;background:#126ca6;border:0;color:white;border-radius:5px;cursor:pointer;margin:8px 12px 8px 0}
#result{white-space:pre-wrap;line-height:1.7}.score{color:#765300}input{margin-right:7px}
</style>
<h1>固定四帧：最低可接受画质标定</h1>
<p>先比较角色眼睛、嘴部、胡须、轮廓和字幕，再为每个版本选择。版本已打乱，暂时隐藏PSNR。请保持浏览器缩放比例一致；需要查看细节时可单独打开图片。页面中的标题和原图来自原始实验预览，分数来自浮点输出，不是对PNG截图重新计算。</p>
<div class="panel"><b>参考原图 · 第2273–2276帧</b><div id="reference"></div></div>
<div id="versions"></div>
<div class="panel"><button id="reveal">显示分数及验收结果</button><button id="save">下载验收记录</button><div id="result">请先判断全部版本，再显示结果。</div></div>
<p>本次只标定该动画GOP、该显示尺寸下的空间画质。没有测试连续播放、真人视频、其他场景或极端丢包；不能据此得到通用PSNR最低要求。即使单GOP通过，也应对整段视频检查低分位和最差片段。</p>
<script>
const items=__DATA__;
let revealed=false,last=null;
function crop(item,isReference){const box=document.createElement('div');box.className='crop '+(isReference?'reference':'reconstruction');box.style.aspectRatio=`${item.width} / ${item.height/3}`;const img=document.createElement('img');img.src=item.file;img.alt=isReference?'参考原图':`重建版本${item.label}`;box.append(img);return box;}
document.getElementById('reference').append(crop(items[0],true));
for(const item of items){const panel=document.createElement('section');panel.className='panel';const heading=document.createElement('h2');heading.textContent=`版本 ${item.label}`;panel.append(heading,crop(item,false));
for(const [value,text] of [['accept','可以接受'],['reject','不能接受'],['unsure','不确定']]){const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name='quality_'+item.label;radio.value=value;label.append(radio,text);panel.append(label);}
const score=document.createElement('div');score.className='score';score.id='score_'+item.label;panel.append(score);document.getElementById('versions').append(panel);}
function calculate(){const answers=items.map(i=>({...i,decision:document.querySelector(`input[name="quality_${i.label}"]:checked`)?.value||'unanswered'}));
const accepted=answers.filter(x=>x.decision==='accept').sort((a,b)=>a.psnr_db-b.psnr_db);
const low=accepted[0];const rejects=answers.filter(x=>x.decision==='reject');
let text=answers.map(x=>`版本${x.label}：${x.psnr_db.toFixed(3)} dB / ${({accept:'可接受',reject:'不可接受',unsure:'不确定',unanswered:'未评分'})[x.decision]}`).join('\\n');
if(answers.some(x=>x.decision==='unanswered'))text+='\\n\\n评分尚未完成，不能确定暂定验收线。';
else if(!low)text+='\\n\\n没有可接受版本。不能给出数值验收线；当前最高分也未被确认可接受。';
else if(rejects.some(x=>x.psnr_db>low.psnr_db))text+='\\n\\n评分不是随PSNR单调改善：有更高分版本仍被拒绝。不能用单一PSNR阈值概括，请同时记录具体失效细节。';
else {text+=`\\n\\n最低被接受的已测版本：${low.psnr_db.toFixed(3)} dB（版本${low.label}）。这是候选验收线，不是精确阈值。`;const below=rejects.filter(x=>x.psnr_db<low.psnr_db).sort((a,b)=>b.psnr_db-a.psnr_db)[0];if(below)text+=`\\n相邻已测范围：${below.psnr_db.toFixed(3)} dB 不接受，${low.psnr_db.toFixed(3)} dB 接受；中间未测试。`;else text+='\\n更低分版本未被明确拒绝，无法确定下界。';}
text+='\\n\\n仅针对当前固定GOP的主观判断。未评分/不确定状态不能作为接受；不证明整段视频达标。';
return {date:new Date().toISOString(),gop:[2273,2274,2275,2276],answers,conclusion:text};}
document.getElementById('reveal').onclick=()=>{last=calculate();revealed=true;for(const i of items)document.getElementById('score_'+i.label).textContent=`PSNR ${i.psnr_db.toFixed(3)} dB；第${i.step}次更新`;document.getElementById('result').textContent=last.conclusion;};
document.getElementById('save').onclick=()=>{last=calculate();const blob=new Blob([JSON.stringify(last,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='psnr_acceptance_result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></html>'''
(out/'画质验收.html').write_text(template.replace('__DATA__',json.dumps(items,ensure_ascii=False)),encoding='utf-8')
(out/'说明.txt').write_text('用浏览器打开画质验收.html，先隐藏分数判断，再揭示结果。下载验收记录并保存回本目录。\n'
    '未重新训练；各版本来自同一个真实模型实验不同更新次数，PSNR取自原metrics.csv。\n'
    '不对图像做额外模糊或锐化；CSS仅裁出已有预览图的原图行和重建行。PNG缩放展示与实际视频播放体验不同。\n',encoding='utf-8')
print(json.dumps([{k:i[k] for k in ('label','step','psnr_db')} for i in items],ensure_ascii=False,indent=2))
print(out.resolve())
