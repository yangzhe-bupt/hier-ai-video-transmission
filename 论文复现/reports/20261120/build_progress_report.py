"""汇总已确认的服务器日志和本地实验记录；不训练或修改模型。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parents[2]
RUN_RECORDS=PROJECT/'reports/run_records'
OUT=PROJECT/'outputs/conquering_report_20261120'
OUT.mkdir(parents=True,exist_ok=True)
ps=[0,.25,.5,.9]
# 阶段I原始MSE来自用户完整终端日志，换算每种子的PSNR后取均值。
baseline_mses=[
    [.01067531]*3,
    [.01265988,.01257466,.01279341],
    [.01848456,.01840597,.01855140],
    [.10862381,.10851340,.11054284],
]
baseline_mean=[statistics.mean(-10*math.log10(v) for v in a) for a in baseline_mses]
baseline_std=[statistics.stdev(-10*math.log10(v) for v in a) for a in baseline_mses]
joint=[19.70459596445774,19.382579313997034,18.87495086699021,16.2264819670092]
joint_std=[0,.010792288355010731,.010778459456453029,.12450740036621342]
decoder=[19.569476534577195,19.24124100566917,18.732310572584645,16.127054637359137]
decoder_std=[0,.017796064882631822,.015513107871692943,.09791722239563602]
bypassed=[19.70459596445774,19.388224212998217,18.871474811390545,15.762712049272015]
records={
    'date':'2026-11-20 Asia/Shanghai',
    'evaluation':'same-video temporal validation; seeds42/43/44; per-seed pooled valid-pixel MSE to PSNR, then arithmetic mean of PSNR',
    'packet_loss_probabilities':ps,
    'stage1_baseline':{'mean_psnr_db':baseline_mean,'std_psnr_db':baseline_std,
        'provenance':'user server log MSE rounded to8 decimals; means reconstructed, approximately accurate',
        'server_dir':'outputs/step13_packet_baseline/20261120_130709_206138'},
    'joint_best_epoch8':{'mean_psnr_db':joint,'std_psnr_db':joint_std,
        'provenance':'user pasted result.json summary; raw evaluation directory not downloaded locally',
        'server_dir':'outputs/step13_stage2_evaluation/20261120_133037_583984'},
    'decoder_only_best_epoch6':{'mean_psnr_db':decoder,'std_psnr_db':decoder_std,
        'provenance':'user pasted result.json summary; raw evaluation directory not downloaded locally',
        'server_dir':'outputs/step13_stage2_evaluation/20261120_134137_981050'},
    'joint_inference_bypass':{'mean_psnr_db':bypassed,'provenance':'user pasted summary; same trained decoder as joint, not independent training'},
    'joint_gain_vs_decoder_db':[a-b for a,b in zip(joint,decoder)],
    'local_evidence':{},
}
for folder in ('20261119_231635_841699','20261120_123347_722927','20261120_131558_052663'):
    path=RUN_RECORDS/folder/'result.json'
    records['local_evidence'][folder]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'result':json.loads(path.read_text(encoding='utf-8'))}
(OUT/'results_and_sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

plt.rcParams.update({'font.size':11})
fig,ax=plt.subplots(figsize=(8,5))
for label,means,std,color in (
    ('Stage I / zero fill',baseline_mean,baseline_std,'#777777'),
    ('Stage II / decoder only (best epoch 6)',decoder,decoder_std,'#d77b17'),
    ('Stage II / recovery + decoder (best epoch 8)',joint,joint_std,'#1574ae')):
    ax.errorbar([p*100 for p in ps],means,yerr=std,marker='o',capsize=4,label=label,color=color)
ax.set(xlabel='Configured packet loss (%)',ylabel='Mean PSNR (dB)',title='Single-video validation / three channel seeds')
ax.grid(alpha=.25);ax.legend(fontsize=9)
fig.text(.5,.015,'Error bars: sample SD across channel seeds; one training seed per model.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'packet_loss_comparison.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4))
gains=records['joint_gain_vs_decoder_db']
ax.bar([f'{p:.0%}' for p in ps],gains,color='#1574ae')
for i,v in enumerate(gains): ax.text(i,v+.002,f'{v:.3f}',ha='center')
ax.set(xlabel='Configured packet loss',ylabel='PSNR difference (dB)',ylim=(0,.18),title='Joint scheme minus independently trained decoder-only scheme')
ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUT/'independent_ablation_gain.png',dpi=180);plt.close(fig)

source=RUN_RECORDS/'20261120_131558_052663/validation.csv'
with source.open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for p in ps:
    subset=[r for r in rows if float(r['p'])==p]
    for ax,key in zip(axes,('mse','psnr_db')):
        ax.plot([int(r['epoch']) for r in subset],[float(r[key]) for r in subset],'o-',label=f'p={p:.0%}')
        ax.set_xlabel('Stage II epoch');ax.set_ylabel(key);ax.grid(alpha=.25);ax.legend()
fig.suptitle('Joint scheme / fixed validation channel seed 42')
fig.tight_layout();fig.savefig(OUT/'stage2_training_curves.png',dpi=180);plt.close(fig)
shutil.copy2(source,OUT/'joint_validation_original.csv')

text='''Conquering复现学习实验阶段汇报
日期：2026-11-20（Asia/Shanghai）

一、汇报开场（可直接使用）
我目前完成了视频语义编解码学习原型：空间和时空Swin编码、公共/个体特征分解、量化、交织及整包擦除、双分支恢复与视频重建。采用两阶段训练，先学习无丢包重建，再冻结发送端训练抗丢包接收端。现有实验验证了链路可学习和抗丢包训练有效，但尚未完成论文数据、通信预算与全部配置的严格对齐。

二、数据与实验条件
单视频531帧，连续四帧组成GOP。训练106个GOP（帧0–423），隔离16帧，验证22个GOP（440–527），尾部3帧舍弃。各GOP互不重叠，但同场景相关性仍然存在。
完整帧等比缩放至256×144，左右补边至256×256，排除新增补边计算损失。视频内部已有的上下黑边仍计入指标。
Adam学习率1e-4，batch1，训练随机种子42。阶段I多GOP30轮3180次更新；阶段II各方案10轮1060次更新。
量化为[-1,1]范围8级模拟，训练阶段I加噪，评估及冻结发送端阶段II舍入。尚无真实3-bit打包或UDP发送。
通信模拟为固定交织、每包16个量化元素，公共/个体分支分别分包，丢失包置零后逆交织。

三、阶段I结构对照（第30轮）
版本                     参数量       训练PSNR      验证PSNR
legacy                  6,974,094     23.974        19.742
common_fig7             6,455,694     23.940        19.716
common_fig7_stage6_pair  7,959,960     23.662        19.768
单位dB；均为单次训练结果，不说明统计稳定差异。
公共提取器图示版本未明显改善质量，减少约7.43%参数；Stage6加深增加约23.3%参数，但验证提升约0.052 dB。
选用较小common_fig7作为后续工作基线。单GOP拟合28.46 dB仅证明固定四帧可拟合，不与多GOP验证直接比较。
量化冻结诊断取消舍入/完全直通仅提高验证约0.030/0.157 dB，不支持把模糊主要归因于当前舍入操作，也不证明量化无损。

四、阶段II训练和比较方法
冻结空间编码器、时空编码器、公共及个体提取器、量化器。训练个体上采样和时空/空间解码器；联合方案另训练公共2D和个体3D恢复网络。
训练GOP随机选择0/30%/60%/90%丢包。联合方案损失=MSE+0.01*MoE辅助损失+0.1*双分支特征恢复损失，恢复目标为量化后的发送特征。
无恢复独立方案从同一个阶段I权重开始，接收端训练设置一致，无恢复损失；两方案属于“恢复模块及其训练目标”的整体对照，不是仅单层算子的对照。
按固定验证种子42、四个丢包条件MSE等权均值选择最佳检查点。联合组第8轮最佳，独立无恢复组第6轮最佳，再用通道seed42/43/44评估。

五、抗丢包结果（三通道种子PSNR算术平均，dB）
丢包率       阶段I置零基线   仅训练接收端    联合恢复方案   联合减独立
0%           19.716          19.569          19.705        +0.135
25%          18.970          19.241          19.383        +0.141
50%          17.333          18.732          18.875        +0.143
90%           9.617          16.127          16.226        +0.099
基线由用户日志中舍入后的MSE重算，表中保留3位小数。联合/独立方案来自用户贴出的完整评估汇总。
联合组90%丢包为16.226±0.125 dB；无恢复独立组16.127±0.098 dB（样本标准差，仅反映通道重复，不代表模型训练不确定性）。

六、可支持的结论
1. 学习原型完成了无丢包和抗丢包两阶段流程；冻结范围、包通道、掩码和梯度已做联合检查。
2. 接收端抗丢包训练带来主要收益：90%丢包基线9.617 dB，独立无恢复训练16.127 dB。
3. 联合恢复方案在本次比较中额外提高约0.10–0.14 dB，尚不足以声称稳定或显著优越。
4. 联合模型推理绕过恢复后90%丢包15.763 dB，比启用低0.464 dB；这与独立训练差值0.099 dB不同，说明不能把推理绕过当作独立贡献分解。
5. 画面主要恢复粗略结构，人物边界、动作和纹理仍模糊。高PSNR改进不代表任务语义或动作识别已恢复。

七、当前限制与论文差距
单视频时间段验证，无跨视频独立测试；每种方案仅一个训练种子，模型选择使用验证seed42。
尚未使用Vimeo-90k训练和HEVC/UVG测试；预处理、batch、模型规模与论文默认设置不同。
C=96的投影位置、默认CBR=0.0078的预算对齐、专家和部分核参数等仍待核实。当前画布口径CBR约0.02148，为元素数比，非实测比特率。
现有实现使用图示支持的结构选择，但不能宣称完整复现。未测MS-SSIM、真实比特流、网络开销或真实UDP链路。
现有恢复网络与辅助目标整体对照略好；不能把0.099 dB差异全部归因于结构，也不能由三个通道种子给出训练显著性结论。

八、建议汇报顺序
问题与论文思路→当前完整路径→数据和两阶段训练→结构对照→抗丢包曲线→独立消融→限制和下一步。
本阶段应定位为“可运行流程与机制验证”，而非达到论文指标。

九、后续优先事项（本次未启动）
先扩大到多个原始视频，按视频划分训练/验证/测试；在资源允许时再引入论文数据集。
保持当前工作基线，重复多个训练种子确认联合恢复增益；避免继续只凭单视频小差异堆叠结构。
补MS-SSIM和一致的视觉对照；正式比较前对齐通信预算。真实比特打包及UDP属于后续独立工程阶段。

十、资料来源与归档状态
本地已核对legacy第30轮、Stage6第30轮及联合阶段II第10轮result/config/validation记录，文件路径和哈希见results_and_sources.json。
common_fig7、阶段I三通道基线、联合最佳及无恢复最佳评估部分来自用户贴出的服务器日志；其原始目录尚未全部下载到本地，不能标为已本地核验。
曲线训练历史直接读取本地联合组validation.csv，其副本保留在材料中。所有图由build_progress_report.py生成，未重新训练或修改权重。
'''
(OUT/'阶段汇报与结果说明.txt').write_text(text,encoding='utf-8')
shutil.copy2(Path(__file__),OUT/'build_progress_report.py')
print(f'材料保存：{OUT.resolve()}')
print('联合方案相对独立无恢复PSNR变化：',records['joint_gain_vs_decoder_db'])
