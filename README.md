# D3 × Who & When Pro：Agent 失败归因评测

使用 [D3 / Decision 3.0](https://huggingface.co/vllm-sr/d3) 对 [Who & When Pro](https://huggingface.co/datasets/Leoxx/whowhen_pro) 的 text、image、image_gui、video 四个分区进行失败归因。

输入一条完整轨迹，独立选择最可能首次引入决定性错误的 **agent、step、error mode**，保存每个选项的概率。三个选择不做一致性修正；概率不等同于已经校准的正确率。本轮是独立选择基线，早期的联合 node 探索代码保留在 `attribution/`，未用于下表主实验。

本仓库仅发布项目代码、测试和 README。模型权重、模型/实验配置（包括 taxonomy.json）、数据集、图片/视频帧、逐条预测、日志、环境和下载清单均不上传。以下结果为本地实验的汇总，不包含数据文件。

## 测试结果

完成日期：2026-10-16。使用官方代码重新评分的结果如下，指标单位均为 **%**。成功指完成推理，不表示预测正确。

| 分区 | 推理成功 / 总数 | 运行错误 | Who | When | What 宏 F1 | All |
|---|---:|---:|---:|---:|---:|---:|
| text | 6,257 / 6,257 | 0 | 69.73 | 73.18 | 22.33 | 25.26 |
| image | 3,506 / 3,899 | 393 | 40.19 | 72.79 | 26.05 | 25.03 |
| image_gui | 1,251 / 1,254 | 3 | 57.50 | 39.47 | 30.58 | 13.31 |
| video | 916 / 916 | 0 | 85.71 | 65.83 | 41.63 | 26.49 |
| **综合（四模态等权）** | — | — | **63.28** | **62.82** | **30.15** | **22.52** |

以上数值由本地 `official_eval/summary.json` 生成，保留两位小数。所有分区均已遍历完毕，但 image/image_gui 存在未成功推理的样本：

- image：385 条未找到可提取的内嵌图片，另外 8 条图片超过服务限制。已核对本地 image.jsonl 的 SHA-256 与下载清单一致，不能通过重下同版本文件解决无图问题。
- image_gui：3 条图片编码大小超过服务限制。
- video：916 条全部完成；82 条发布轨迹没有观测帧，明确作为仅文本证据处理。

官方 leaderboard 默认跳过带 `error` 的记录，因此主表 image/image_gui 的分数只覆盖成功输出。补充将全部失败计错后，四模态综合指标为 **Who 63.28、When 61.17、What 29.85、All 21.90**。这项补充不是官方默认过滤口径。

## 官方评分规则

直接使用 [ag2ai/whowhen_pro](https://github.com/ag2ai/whowhen_pro) 的 `whowhen_eval.score.score` 和 `whowhen_eval.leaderboard`，固定提交：

```text
14369dcb079c7cdf86ed2924145d5a28ff5c73b2
```

- **Who**：各多 agent 框架内计算 agent 准确率，再对框架等权平均。官方将 smolagents、alfagent、mathchat、pixelcraft、eva、openai_cua、agentoccam、gemini 排除在 Who 分母之外；它们仍参与其他指标。
- **When**：各框架内计算步骤匹配准确率，再对框架等权平均。
- **What**：在该模态全部有效记录上计算错误类型宏 F1，只对有真实标签支持的类别平均，不固定除以17。四个分区分别出现14、10、7、5种错误类型。
- **All**：逐条要求 agent、step、mode 三个官方判定同时命中，再对框架等权平均。
- **综合**：四个模态的对应指标等权平均。

部分 ground_truth 含 accepted_predictions，官方逐轴接受替代标签；但 What 宏 F1 仍使用标准 ground_truth.mode。错误类型命中率与 What 宏 F1 不是同一个指标。

这是一份**使用官方评分代码重评分的本地实验**，不是官方榜单提交，也不是严格复现官方输入协议。D3 使用受限选项选择，而论文中的 LLM 通常自由生成；输入渲染、抽帧和部分候选空间也有差别，不应直接把分数差当作公平的模型优劣结论。

## 输入与实验方法

| 分区 | 输入 | 步骤与特别处理 |
|---|---|---|
| text | 完整用户问题及文本轨迹 | 保留执行内容，独立选择 agent/step/mode |
| image | 完整文本与实际内嵌图片 | 按图片哈希去重，保留图片与步骤的对应关系；无图显式报错 |
| image_gui | 完整 GUI 轨迹与截图 | 保留截图对应步骤；超出像素限制的截图缩放，仍可能遇到编码字节数限制 |
| video | 完整文本与工具已观测帧 | 最多32帧，按帧引用顺序均匀选择、包含首尾，保持比例缩至448×448内，保留时间戳及所属步骤 |

video 发布数据包含工具观测帧，不是原始 MP4；不根据答案或错误标签选帧。无帧记录标为 `text_only_no_frames_in_release`。所有适配器均排除 ground_truth、注入标记等标签元数据，任务参考答案不作为模型输入。

重评分只转换已有预测的字段/坐标，不重新调用模型，不根据标签修改预测：MathChat 的 R.P 转为 2R+P；DVD 原始消息下标减2；EVA 转为 (下标−2)//2，assistant/tool 归入同一官方步骤；GUI 单 agent 标签转换为官方固定名称。MacNet 旧实验将 critic/rewriter 合并为1.0，重评分保留实际输出，无法事后恢复成官方的两个独立候选。

## 代码结构

```text
attribution/       早期 node/独立选择实验及测试
v0_test/           文本全量独立选择实验
v0_image/          图片轨迹适配、运行器和测试
v0_image_gui/      GUI 截图适配、运行器和测试
v0_video/          视频观测帧适配、运行器和测试
official_eval/     官方代码重评分适配器及测试
scripts/          根据外部数据集生成本地 taxonomy 的辅助程序
d3_deploy/         模型下载、按物理 GPU 部署、状态检查与停止脚本
```

第三方 D3 实现和官方评测源码不随本仓库重复发布；通过原始仓库或模型页单独获取。`references/`、`whowhen_pro/`、`d3_deploy/model/` 和 `results/` 均为本地生成/下载内容。

## 环境与外部文件准备

当前实验使用 conda 环境 `d3_v1`。新机器请根据 D3 模型页准备兼容的 PyTorch、Transformers 和 D3 依赖；实验还使用 Pillow、PyYAML，下载脚本使用 requests。仓库中的部分 shell 包装器沿用原机器的 conda 绝对路径，跨机器推荐直接用当前环境的 `python` 执行下述程序。

```bash
conda activate d3_v1
python -m pip install pillow pyyaml requests huggingface_hub

# 数据与模型独立下载，不属于上传的代码文件。
hf download Leoxx/whowhen_pro --repo-type dataset \
  --revision 0bd196c8a040841c4ae167ab33cc8151de246f1f --local-dir whowhen_pro
hf download vllm-sr/d3 --local-dir d3_deploy/model

# 从外部数据集的分类表生成各实验目录需要的 taxonomy.json。
python scripts/prepare_taxonomy.py

# 官方重评分代码；固定到本次使用的提交。
git clone https://github.com/ag2ai/whowhen_pro.git references/whowhen_pro_official
git -C references/whowhen_pro_official checkout 14369dcb079c7cdf86ed2924145d5a28ff5c73b2
```

D3 下载命令默认使用模型仓库当时的 main；若需严格复现权重，请另行固定和校验模型 revision，本代码仓库不发布模型 manifest。

## 指定 GPU 部署

```bash
# gpu 是 nvidia-smi 显示的物理卡号。模型文件和 conda 环境可复用；各卡独立加载一份显存权重。
python d3_deploy/deploy.py start --gpu 0 --port 8899 --python "$(command -v python)"
python d3_deploy/deploy.py start --gpu 1 --port 8900 --python "$(command -v python)"
python d3_deploy/deploy.py start --gpu 2 --port 8901 --python "$(command -v python)"
python d3_deploy/deploy.py start --gpu 3 --port 8902 --python "$(command -v python)"

python d3_deploy/deploy.py status --gpu 3 --port 8902
python d3_deploy/deploy.py stop --gpu 3 --port 8902
```

服务接口为 `POST /v1/systemone`，三个 typed choice 问题一次请求返回。当前部署管理器按至少约55GB空闲显存预检，长轨迹与多图输入有额外显存需求；本轮使用80GB级GPU。

## 小样本与全量运行

先确保对应端口服务已就绪。四个实验默认端口依次为8899、8900、8901、8902。

```bash
# 文本：先5条，再全量；正式全量使用新目录以保留小样本。
python v0_test/run_full.py --limit 5 --output-dir v0_test/results/smoke
python v0_test/run_full.py --output-dir v0_test/results

# 多模态：每个框架2条小样本。
python v0_image/run.py --per-framework 2 --output-dir v0_image/results/smoke
python v0_image_gui/run.py --per-framework 2 --output-dir v0_image_gui/results/smoke
python v0_video/run.py --per-framework 2 --output-dir v0_video/results/smoke

# 全量。
python v0_image/run.py --all --output-dir v0_image/results/full
python v0_image_gui/run.py --all --output-dir v0_image_gui/results/full
python v0_video/run.py --all --output-dir v0_video/results/full
```

以上命令在前台运行；需要长期运行时自行放入 tmux 或后台任务。image/image_gui/video 中断后，用完全相同命令加 `--resume` 继续，跳过成功项并重试错误项；文本运行器自动跳过成功记录，重试错误使用 `--retry-errors`。已有目录的配置/代码指纹必须一致，修改实验后请使用新输出目录。

运行器在本地输出 `predictions.jsonl`、`summary.json`、`progress.json`、manifest 等，部分实验同时保存去除图片字节的审计请求；这些输出文件不上传。`complete` 表示遍历完成，不表示每条均推理成功。

```bash
watch -n 10 cat v0_video/results/full/progress.json
```

## 重算官方指标

需先生成/恢复四个默认路径下的 predictions.jsonl；仓库不包含预测数据，因此不能在空结果目录直接复现上表。

```bash
python official_eval/rescore.py
PYTHONPATH=references/whowhen_pro_official \
  python -m whowhen_eval.leaderboard --results official_eval/results
python -m unittest discover -s official_eval -p 'test_*.py' -v
```

重评分会生成 `official_eval/summary.json`、`summary.csv` 和 `results/d3/*.jsonl`，保留原实验预测不变。summary 包含输入预测文件哈希、逐框架指标、成功样本口径与失败计错口径。

## 参考

- [D3 模型与官方使用说明](https://huggingface.co/vllm-sr/d3)
- [Who & When Pro 数据集](https://huggingface.co/datasets/Leoxx/whowhen_pro)
- [Who & When Pro 官方评测仓库](https://github.com/ag2ai/whowhen_pro)
- [Who & When Pro 论文](https://arxiv.org/abs/2607.09996)
- [Jev 文本归因实验](https://github.com/TokenTrim/jev-agent-failure-benchmark)：参考其独立 typed choice 的思路；本仓库实际运行模型是 D3，不是 Jev。

数据集授权为 CC-BY-4.0，D3 模型页标注 Apache-2.0；使用外部数据、模型和评测代码时请遵守各自授权。
