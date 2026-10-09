# 文本 LoRA：Qwen3.5-4B + ms-swift

按照[分享对话](https://chatgpt.com/share/6ac5c263-3e08-83ec-8240-d3df08cce17d)在当前仓库落地文本微调项目。环境检查、10 条原模型基线、5 条教学样本审阅与 30/6 数据扩充已完成。2026-10-08 已通过真实 3 步 LoRA 试跑和 3 轮教学训练；训练、固定测试和对照入口在 `04_lora_training.ipynb`。

## 目录和当前进度

```text
text-lora/
├── .python-version                 # 3.12.13，与本仓库已有环境一致
├── pyproject.toml                  # uv 依赖声明
├── configs/                       # 本地模型路径、固定推理与训练参数
├── notebooks/
│   ├── 01_environment.ipynb        # 项目 kernel / GPU / 依赖 / tokenizer 检查
│   ├── 02_base_inference.ipynb     # 单条推理、10 条固定测试与原模型基线
│   ├── 03_dataset_preparation.ipynb # 固定标注写法、5 条样本、检查与导出
│   ├── 04_lora_training.ipynb      # 数据监督、3 步试跑、3 轮训练与固定测试对照
│   ├── 05_data_iteration.ipynb     # 第二轮 50/12 数据与原模型/两轮 LoRA 对照
│   └── 06_suspense_style_lora.ipynb # 面向 TTS 的悬疑措辞、节奏与演绎风格
├── scripts/
│   ├── check_environment.py        # 环境检查
│   ├── run_inference.py            # 基线与 adapter 推理共用入口
│   ├── prepare_training_data.py    # 原创语料导出、来源/格式/模板检查
│   ├── train_lora.py               # smoke/full 训练与 checkpoint 核验
│   ├── compare_inference.py        # 配置核对、按 ID 生成输出对照
│   ├── compare_iterations.py       # 原模型、第一轮、第二轮三组对照
│   ├── run_second_iteration.py     # 第二轮一键检查、训练与评估
│   ├── view_second_iteration.py    # 独立只读状态与报告入口
│   ├── prepare_style_data.py      # 新风格语料、目标约束与 CPU 模板检查
│   ├── run_style_iteration.py     # 8 阶段风格实验，完整结果自动复用
│   ├── export_style_results.py    # 无标签朗读正文与角色/语气清单
│   └── view_style_results.py      # 独立只读样稿、状态与结果查看
├── eval/                          # 固定测试提示词与人工检查标准
├── datasets/                      # SFT 数据约定与 annotation_guide.md 标注指南
└── outputs/                        # 后续 adapter、checkpoint、评估结果
```

`.venv/` 和 `uv.lock` 由首次手动安装依赖时生成。`uv.lock` 应提交到 Git，`.venv/`、真实数据和训练结果已忽略。当前环境已安装依赖，命令行的 Python、依赖导入、CUDA/BF16 和离线 tokenizer 检查全部通过；`text-lora` kernel 已注册。

本项目单独管理 uv 环境。JupyterLab 继续使用全局 `uv tool` 环境，Notebook 使用 `text-lora/.venv` 的 kernel。原有 `Qwen3.5-4B-vLLM/` 与 `Qwen3.5-4B-LoRA和SwanLab/` 属于其他实验目录，新的 ms-swift 工作流统一放这里。图片 LoRA 按分享对话的顺序留待后续建立独立项目。

## 已确认的本地模型

2026-10-07 检查了模型配置、tokenizer 文件和权重索引引用的分片；两套模型的文件均存在且非空。2026-10-08 默认 Qwen3.5-4B 已完成真实基线推理及文本 LoRA 教学训练；备用模型尚未执行权重加载。

| 配置名称 | 本地目录 | 用途 |
| --- | --- | --- |
| `qwen3_5_4b`（默认） | `/home/muyi086/hf-mirror/Qwen/Qwen3.5-4B` | 分享对话选定的主路线，`Qwen3_5ForConditionalGeneration` |
| `qwen3_4b_instruct` | `/home/muyi086/hf-mirror/Qwen/Qwen3-4B-Instruct-2507` | 备用文本模型，`Qwen3ForCausalLM` |

`configs/models.toml` 用 `~` 表示用户目录，迁移机器时修改此文件即可。Qwen3 与 Qwen3.5 是不同模型；默认沿用对话中的 Qwen3.5。环境检查脚本和 `01_environment.ipynb` 不加载权重；`02_base_inference.ipynb` 的真实推理 Cell 会加载本地权重到 GPU。

现在就能运行仅依赖 Python 标准库的文件检查：

```bash
cd /home/muyi086/github/test_LLM/text-lora
python3 scripts/check_environment.py --local-only
python3 scripts/check_environment.py --local-only --model qwen3_4b_instruct
```

## pyproject.toml 的建议

以下是依赖声明采用的起点，实际解析版本以 `uv.lock` 为准。当前环境检查已通过，尚未执行 CUDA 训练验证：

| 项目 | 建议 |
| --- | --- |
| Python | 3.12.13；项目限定 3.12 系列 |
| PyTorch / torchvision | 2.11.0 / 0.26.0，显式使用 cu130 CUDA wheel 源 |
| ms-swift | 4.5.3 |
| Transformers | `>=5.2,<6`，由首次 uv 解析锁定具体版本 |
| PEFT / datasets | `>=0.18,<0.20` / `>=4.0,<4.8.5` |
| Notebook | ipykernel + ipywidgets，全局 JupyterLab 单独管理 |

分享对话建议 Python 3.11；这里结合已安装的 Python 3.12.13 和 [ms-swift 当前官方推荐](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html)采用 3.12。ms-swift 4.5.3 已在 [PyPI](https://pypi.org/project/ms-swift/4.5.3/) 发布。Qwen3.5 的[官方训练示例](https://swift.readthedocs.io/en/v4.0/BestPractices/Qwen3_5-Best-Practice.html)使用 Transformers 5.x，不能把模型 config 中的 `4.57.0.dev0` 当作当前安装下限。

CUDA wheel 的[显式源配置](https://docs.astral.sh/uv/guides/integration/pytorch/)沿用现有仓库的 cu130 版本路线，使用 PyTorch 官方源。普通包保留阿里云镜像；下载方式可按你的需要修改。运行时以 `torch.cuda.is_available()` 和 BF16 检查为准，`nvidia-smi` 成功不代表项目的 PyTorch 已可用。

第一阶段的基础清单不包含 Flash Attention、flash-linear-attention、causal-conv1d、bitsandbytes、DeepSpeed、vLLM 或 SwanLab。Qwen3.5 后续若需要线性注意力加速算子，单独核对 Torch/CUDA/Python 的组合再添加。当前基础清单通过环境检查也不等于已证明训练性能或显存足够。

## 你稍后手动执行的安装与检查

下面的安装及 kernel 注册命令仅供你手动执行，项目代码不会调用它们。

```bash
cd /home/muyi086/github/test_LLM/text-lora

# 安装项目依赖，并首次生成 .venv 和 uv.lock。
# --no-python-downloads 避免隐式下载 Python；本机已有 3.12.13。
uv sync --no-python-downloads

# 依赖准备好后，禁止运行命令自动同步或安装包。
uv run --no-sync --no-python-downloads python scripts/check_environment.py

# 手动注册项目 kernel，供全局 JupyterLab 使用。
uv run --no-sync --no-python-downloads python -m ipykernel install \
  --user --name text-lora --display-name "Python (Text LoRA - Qwen)"
```

全局 JupyterLab 尚未准备好时，可手动执行 `uv tool install jupyterlab`。已经安装则在项目目录检查 kernel，再使用全局入口启动：

```bash
uv run --no-sync --no-python-downloads jupyter kernelspec list
jupyter-lab
```

`uv tool install jupyterlab` 对外暴露 `jupyter-lab` 等 JupyterLab 自身的入口，依赖包 `jupyter-core` 提供的 `jupyter` 不会自动暴露到全局 PATH，详见 [uv 工具入口规则](https://docs.astral.sh/uv/concepts/tools/#installing-executables-from-additional-packages)。因此 `jupyter-lab` 可以正常启动，而裸写 `jupyter kernelspec list` 可能提示命令不存在。上面的 `uv run --no-sync` 使用项目环境中已有的 `jupyter`，无需额外安装系统包。

`kernelspec list` 应包含 `text-lora`，目录为 `~/.local/share/jupyter/kernels/text-lora`。全局 JupyterLab 同样能读取该用户目录中的 kernel。若服务器已经在其他终端运行，继续使用已打开的页面，刷新页面后选择 kernel 即可。

打开 `notebooks/01_environment.ipynb`，选择 `Python (Text LoRA - Qwen)`，按顺序运行全部 Cell。Notebook 的 Python/kernel 检查应通过，GPU 应显示 RTX 4070 Ti SUPER，CUDA 和 BF16 检查应通过；最终 tokenizer 检查只加载配置和 tokenizer，不加载 4B 权重。

这里的 `--no-sync` 有意保留，因为普通 `uv run` 会自动解析与同步依赖，详见 [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)。

## Notebook 中的 IProgress 警告

若训练依赖 Cell 显示 `TqdmWarning: IProgress not found`，但最后是 `[PASS]`，表示依赖导入成功，`tqdm.auto` 因缺少 Notebook widgets 回退到文本进度条。该警告不影响已完成的环境检查，也不影响训练计算。

此前该警告对应项目 kernel 缺少 `ipywidgets`，全局 JupyterLab 的 tool 环境缺少 `jupyterlab_widgets`；现已按此流程完成配置。这两个环境分开维护，需要分别配置 Python widgets 与网页显示扩展，详见 [ipywidgets 官方安装说明](https://ipywidgets.readthedocs.io/en/stable/user_install.html#installing-in-jupyterlab)。

项目 `pyproject.toml` 已补充 `ipywidgets>=8.1,<9`。如果希望使用图形进度条，手动执行以下命令；这些安装命令尚未由 agent 执行：

```bash
cd /home/muyi086/github/test_LLM/text-lora

# 同步新增的 kernel 依赖，并更新项目锁文件。
uv sync --no-python-downloads

# 为全局 JupyterLab 添加网页 widgets 扩展，保留当前 JupyterLab 版本。
uv tool install --no-python-downloads \
  --with "jupyterlab-widgets>=3,<4" "jupyterlab==4.6.4"
```

随后保存 Notebook，重启全局 JupyterLab 服务及 Notebook kernel，再按顺序运行 Cell。新包安装后应重启 kernel，因为 `tqdm` 会缓存 widgets 是否可用。可在 Notebook 新建临时 Cell 验证：

```python
from ipywidgets import IntProgress
from IPython.display import display

display(IntProgress(value=1, min=0, max=3, description="Widgets"))
```

显示图形进度条表示两侧配置均可用；若仅显示 `IntProgress(...)` 文字，检查全局 JupyterLab 的 widgets 扩展及服务是否已重启。

## 第二阶段：先保存原模型基线

在 JupyterLab 打开 `notebooks/02_base_inference.ipynb`，选择项目 kernel 后逐 Cell 执行：

1. 预览 `eval/prompts.jsonl` 的 10 个固定输入：8 个小说改写任务、2 个普通文本任务。
2. 执行 dry run，确认配置和本地文件。
3. 运行第一条真实推理，确认当前模型能完成生成。
4. 运行完整 10 条，保存基线并填写输出目录下的 `review.md`。

推理使用已安装的 ms-swift `TransformersEngine`，参数按[官方 Python 推理接口](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html#using-python)和本地 4.5.3 代码核对。只使用本地模型路径，启用 Hugging Face 离线模式；BF16、单条、SDPA 路线不新增依赖、不使用量化。默认模型已通过 Notebook 完成真实权重加载和 10 条生成。

也可以在终端分步执行：

```bash
cd /home/muyi086/github/test_LLM/text-lora
uv run --no-sync --no-python-downloads python scripts/run_inference.py --dry-run
uv run --no-sync --no-python-downloads python scripts/run_inference.py --limit 1
uv run --no-sync --no-python-downloads python scripts/run_inference.py
```

第一条 dry run 不加载模型、不创建输出目录。后两条会真实加载模型，并分别创建 `outputs/base-时间戳/`；每次运行结束后，子进程退出并释放模型显存。运行前结束其他占用显存的模型服务。

完整基线目录包含：

- `results.jsonl`：输入、模型输出、token 用量、耗时与 `finish_reason`。`length` 代表触及输出上限，需人工检查是否截断。
- `metadata.json`：模型路径、配置哈希、依赖版本、提示词哈希、推理设置、运行状态和完成条数。
- `review.md`：Notebook 创建的人工观察清单。命令行仅保存前两个文件。

输出逐条写入并刷新；生成中报错时保留已完成记录。以 `status=complete` 且 `completed_count=10` 的结果作为完整基线。脚本拒绝覆盖已有运行目录。

保持提示词和 `configs/inference.toml` 的设置固定。未来可给同一个脚本传入 `--adapter <本地 checkpoint>` 生成 LoRA 输出，核对两次记录中的提示词哈希、模型和推理设置后再比较。这里的“原模型”指当前 post-trained Qwen3.5-4B，和名为 Qwen3.5-4B-Base 的另一个模型无关。

先记录格式遵循、原文事实保留、说话者归属和新增情节等表现。不要要求所有输出都达到目标格式，也不要将这些固定测试片段加入后续训练集。

## 第三阶段：教学数据准备（已完成）

先阅读 [datasets/annotation_guide.md](datasets/annotation_guide.md)，再打开 `notebooks/03_dataset_preparation.ipynb`。5 条演绎稿均已由 agent 编写完整，查看 [datasets/pilot_review.md](datasets/pilot_review.md) 可审阅样本和检查结果；检查通过并同意采用后导出教学样本。agent 负责数据编写、修订、技术检查与训练入口，你主要负责监督和决策。该 Notebook 离线读取 processor，不加载模型权重、不安装依赖、不启动训练。

已完成的基线是 `outputs/base-20261008T012651_918242Z/`。用户已执行 Notebook 03 并批准 5 条样本，导出至 `datasets/processed/pilot-20261008T120902_009657Z/`。它完成写法统一、格式检查、实际模板长度检查和导出，没有训练模型。原始推理结果、固定测试输入和推理配置继续保留。

完整原创语料在 [datasets/teaching_samples.json](datasets/teaching_samples.json)：30 条训练、6 条独立验证，训练中复用已批准的 5 条；其余样本由 agent 编写并核对，不代填用户审阅状态。检查与来源记录在 `datasets/processed/teaching-v1/`。[扩充核对记录](datasets/expanded_review.md)说明分配、约束和复核边界。

## 第四阶段：按 1、2、3 查看训练与效果

打开 [notebooks/04_lora_training.ipynb](notebooks/04_lora_training.ipynb)，继续选择项目 kernel，从上到下运行：

1. **数据监督**：自动导出 30/6 JSONL，检查来源分离、重复、固定测试隔离、完整模板长度，并展开查看原文与答案；不需要你补写样本。
2. **3 步试跑**：查看有限 loss、checkpoint 和显存记录。本轮已真实通过。默认复用 `outputs/latest_teaching_run.json` 指向的完整且数据/参数匹配的结果。
3. **3 轮训练与 10 条对照**：正式训练从原模型重新初始化 LoRA，按独立验证 loss 选 checkpoint，然后用固定推理入口加载 adapter，自动核对模型/提示词/依赖/解码设置并按 ID 对齐。你主要审阅效果与决定下一轮方向。

默认 `REUSE_COMPLETED_RUNS=True` 避免重复训练。设为 `False` 才重新实验；新运行创建新目录。脚本不安装依赖、不上传结果，子进程结束后释放权重显存。Notebook 03 和已批准的 pilot 无需重做。

训练设置保存在 [configs/training.toml](configs/training.toml)，本次实际通过的是：

```text
BF16 / LoRA rank=8 / alpha=32 / target_modules=all-linear
max_length=512 / batch_size=1 / gradient_accumulation_steps=8
gradient_checkpointing=true
freeze_vit=true / freeze_aligner=true
learning_rate=1e-4 / num_train_epochs=3 / seed=data_seed=42
enable_thinking=false / loss_scale=default+ignore_empty_think
```

完整模板最长 232 tokens；超限直接报错，不静默截断。训练仅计算 assistant 正文等目标 token 的 loss，忽略 system/user 与模板添加的空思考前缀。原始推理输出中的空 think 块继续单独计数，格式评分只去除开头空块，非空思考内容不会被隐藏。

RTX 4070 Ti SUPER 16 GB 实测：3 步试跑峰值 PyTorch reserved 9.902 GiB；3 轮训练 12 步、reserved 10.137 GiB，最终验证 loss 0.1063。这个统计只包含当前进程的 PyTorch 分配；36 条短样本和 10 条测试只支撑教学流程验证。改变长度或数据量后需重新实测。

[本轮 agent 评审](eval/lora_review_20261008.md)已逐条记录结果：允许标签后空格的正文双标签格式由 3/8 提升到 8/8，但 5 条仍有动作遗漏、台词归属或重复问题；两条普通控制未观察到明显退化。原始输出仍带模板空 think 块，格式指标只检查去空前缀后的正文。

命令行入口（都使用当前环境，不同步依赖）：

```bash
cd /home/muyi086/github/test_LLM/text-lora
uv run --no-sync --no-python-downloads python scripts/prepare_training_data.py --check-tokens
uv run --no-sync --no-python-downloads python scripts/train_lora.py --mode smoke --dry-run
uv run --no-sync --no-python-downloads python scripts/train_lora.py --mode smoke
uv run --no-sync --no-python-downloads python scripts/train_lora.py --mode full
```

试跑修复了 ms-swift 默认 `persistent_workers=True` 与本项目 `num_workers=0` 的冲突，显式设置为 `False`。保存运行内数据快照、所有可训练参数名、loss 历史、峰值显存、状态与选中 adapter；失败也保留 metadata。checkpoint 包含 adapter 权重以及恢复用的训练状态，推理直接加载 adapter，不合并或改写原模型。

参数按 [ms-swift 4.5.3 文档](https://swift.readthedocs.io/en/v4.5/Instruction/Command-line-parameters.html)及本地安装源码核对。该版本入口仍使用 `torch_dtype`；Transformers 的弃用提示不等于训练失败。`decord` 是视频依赖提醒，本次纯文本流程已通过，不需要因此安装新包。

## 读完 04 后：第二轮数据改进

第一轮有 5 条演绎稿仍存在动作遗漏、直接台词归属或重复问题。下一步打开 [notebooks/05_data_iteration.ipynb](notebooks/05_data_iteration.ipynb)：步骤 1 监督新增数据，步骤 2 一个 Cell 自动训练与评估，步骤 3 审阅原模型/第一轮/第二轮的两份对照。用户继续以监督和决策为主，agent 负责具体数据与执行工作。

[第二轮方案与核对记录](datasets/iteration2_review.md)已落实为 50 条训练、12 条验证，完整模板最长 235 tokens；另写并固定 10 条未见测试及参考写法。2026-10-09 用户已完成 3 步试跑、21 步正式训练、40 条生成与两份对照。[第二轮 agent 内容评审](eval/lora_review_20261009.md)已逐条核对：原有 8 条改写由第一轮 3/8 到第二轮 7/8，新 8 条由 5/8 到 8/8；仍有动作遗漏与普通解释细节不足，建议先用更复杂的独立文本验收。数据独立保存在 teaching-v2，第一轮数据、Notebook、checkpoint 与复用记录保留。

入口 `scripts/run_second_iteration.py --dry-run` 可只检查计划；执行时自动完成试跑、正式训练、原测试第二轮生成、新测试三组生成与报告。按验证 loss 选模型，测试参考不进入模型输入或训练。成功记录到 `outputs/latest_iteration2.json`，再次运行复用匹配的完整结果。

步骤 2 包含 9 个阶段，中间训练通过后还会继续生成和报告；以最后的 `[COMPLETE]` 和 Cell `[*]` 消失作为整轮完成标志。步骤 3 使用 `view_second_iteration.py` 独立读取状态和已有报告，重新打开 Notebook 后可直接执行。执行器显示阶段/逐条推理进度，并拦截并发启动，避免误触新的 GPU 任务。

## 第六阶段：明确恐惧悬疑小说的 TTS 改写目标

用户已明确希望训练一种让恐惧悬疑小说更适合有氛围感朗读的 LoRA。前两轮的保守样本主要验证角色、标签与事实保留，不能用它们的内容通过率代替风格和实际听感验收。

新入口为 [06_suspense_style_lora.ipynb](notebooks/06_suspense_style_lora.ipynb)。agent 已另写 30 条训练、10 条验证和 10 条独立测试，使用 [悬疑风格约定](datasets/suspense_style_guide.md)：适度调整措辞、断句和语气，突出已有线索，保留动作和直接台词，不新增情节。先看 [五条样稿摘要](datasets/suspense_samples_review.md)；完整语料在 `datasets/suspense_samples_v1.json`，独立导出为 teaching-v3，最长完整模板 267 tokens。

Notebook 的三个代码 Cell 各自独立：步骤 1 只读监督样稿；步骤 2 自动执行或复用 8 阶段实验；步骤 3 只读查看摘要、agent 评审和三组内容对照。三组为原模型、第二轮保守 LoRA、新风格 LoRA，使用同一份新风格指令及独立的 `configs/style_inference.toml`；本次输出上限为 768，不改前两轮固定配置。checkpoint 仍按独立验证 loss 选择。

运行结果记录在 `outputs/latest_suspense_run.json`。导出各小说候选的 `speech.txt`（无角色/语气标记）和 `performance.json`（逐句角色、语气和正文）；具体 TTS 确定后再映射到语音控制。本阶段先检查文本，音频氛围需实际试听。数据、脚本、实验与逐条核对由 agent 负责，用户监督改写强度与决定是否采用。

2026-10-09 agent 已完成第一版风格实验：3 步试跑、12 步正式训练、30 条生成、对照和朗读候选导出。[逐条评审](eval/suspense_review_20261009.md)确认学到了短句和语气变化，也记录了复杂场景动作遗漏、一次新增细节、平稳场景过度渲染及普通解释错误。Notebook 06 已在独立 kernel 验证查看与结果复用，并保存真实输出；步骤 2 已执行完成，用户可直接监督步骤 1 与步骤 3。

## 第七阶段：方案 A，VoxCPM2 可控克隆

用户已选择“悬疑演绎稿 LoRA + VoxCPM2 可控克隆”。新监督入口为 [07_voxcpm_style_review.ipynb](notebooks/07_voxcpm_style_review.ipynb)：步骤 1 查看完成状态，步骤 2 播放真实音频对照，步骤 3 查看本轮评审并保存试听意见。三个 Cell 可独立运行，只查看本地结果，不启动模型服务、不执行训练，不需要重跑 05 或 06。

`[角色][语气]` 是中间稿：`scripts/voxcpm_adapter.py` 解析角色与语气，把角色映射到参考音色，把语气映射到 [控制词典](configs/voxcpm_delivery.json)。HTTP `text` 仅含正文；可控模式使用 `clone_mode=controllable` 和 `control_instruction`，不传 `prompt_text`、`style_prompt` 或非语言标签。更克制模式合并相邻同说话者，旁白统一自然平稳；其他强度按角色与语气合并。跨请求停顿由拼接器处理，平静段落不随强度选择变成紧张段落。

第一轮 7 份真实试听由两条同旁白的“原文平读 / 改写后平读 / 可控演绎”和一条三角色示范组成，各轮记录保存在 `outputs/voxcpm-*`，最新一轮见 `outputs/latest_voxcpm_audition.json`。素材来自已有风格 LoRA 的已审阅稿，平稳片段的语气由 agent 修订；它们用于验证声音控制方向，不代表第二版新模型的声音结果。用户现有三份合成音色暂作参考。每次请求、音频参数和稿件来源保留；WAV 结构与非静音检查不能代替听感和漏读检查。

新语料 [suspense_voxcpm_v2.json](datasets/suspense_voxcpm_v2.json) 为 42 训练 / 14 验证，独立导出 teaching-v4；保留旧数据，修订四处目标，并加入长动作链、对象和位置、多人对白、平稳反例及普通任务。另留 10 条全新测试，见 `eval/voxcpm_benchmark_v2.json`，推理输入另存且不含参考答案。新训练指针为 `outputs/latest_voxcpm_style_run.json`，与第一版分开。

agent 维护和执行的命令如下，用户监督无需重复执行：

```bash
.venv/bin/python scripts/prepare_style_data.py --corpus datasets/suspense_voxcpm_v2.json --check-tokens
.venv/bin/python scripts/run_style_iteration.py --corpus datasets/suspense_voxcpm_v2.json
.venv/bin/python scripts/run_voxcpm_audition.py
.venv/bin/python scripts/apply_voxcpm_feedback.py
```

完整且输入一致的训练或试听会复用结果。两个试听命令默认仅复用手动启动的 VoxCPM2 服务；服务不存在时提示手动 `bash start.sh`，不会自动占用端口。初轮`run_voxcpm_audition.py`只有显式 `--start-service` 才允许临时启动本机 API，结束后关闭本次进程；反馈处理器始终不启动服务。Notebook 07 始终不启动服务。真实 GPU 操作需与本地语音服务共享 GPU 文件锁，避免训练与语音推理同时占显存。

### 按试听意见调整声音控制

第二轮曾采用轻演绎与140ms追加停顿，见[第二轮处理记录](eval/voxcpm_feedback_review_20261009.md)。该轮之后用户反馈仍然情绪过重、句间过渡不自然、角色接话太快，因此第三轮继续修订。

第三轮的`mild`旁白曾统一自然平稳，不因一行“紧绷”扩展为整段紧张。两条试听稿的标点由agent恢复句内联系，正文去除标点后的字符序列及对白归属、重复次数均核对。修订稿另存[voxcpm_audition_v2.json](eval/voxcpm_audition_v2.json)，不覆盖旧模型输出，不用于训练。

角色切换按总低幅度间隔控制：旁白接对白400ms、对白接旁白500ms、不同角色接话650ms；先测量保守裁切后的片段头尾空白，仅补齐不足。边缘处理保留80ms保护区，单边最多裁1200ms，句内停顿不裁切或变速。参考音色、模型、种子沿用，本轮没有重训文本LoRA。

第三轮的实际时长、请求和边界测量见[第三轮处理记录](eval/voxcpm_feedback_review_v3_20261009.md)。

第三轮之后的意见认可句间节奏的改善，但指出旁白太平、角色音量不一致且对白仍夸张。第四轮稿件另存[voxcpm_audition_v3.json](eval/voxcpm_audition_v3.json)，正文与标点沿用，仅恢复旁白的语气变化标签。当前`mild`对包含情绪标签的连续旁白使用“细微起伏、整体平稳”，纯平静稿仍使用平静指令；对白改用日常交谈指令，取消易放大气声的轻声要求，疑问通过原问号保留。

反馈试听处理器通过已有FFmpeg测量每段输入LUFS及真峰值，用固定增益匹配到-23 LUFS；原文要求低声的洛洵目标-24 LUFS。真峰值保护为-2 dBTP，增益与受限情况记录，保留原始与匹配后的片段，固定响度增益保留句内动态、样本数量和语速；第五轮另增加独立的语速处理。已认可的标点及400/500/650ms接话目标沿用。

第四轮的实际响度、请求和边界测量见[第四轮处理记录](eval/voxcpm_feedback_review_v4_20261009.md)。

第四轮之后的意见认可整体改善，希望语速再缓一些，并用更多新例子检查情绪变化。第五轮设置`mild.tempo=0.94`，以保持声线的atempo处理轻微放缓发声片段，之后重新匹配响度，再独立补齐原有400/500/650ms角色边界。TTS控制指令与种子沿用；每段保留原始、tempo和leveled文件及实际帧数。

新试听稿[voxcpm_audition_v4.json](eval/voxcpm_audition_v4.json)含一份同稿速度参考和五份agent编写审阅的新例子：照片中的不确定线索、门外迟疑问答、略急后缓和、纸条追问、平稳清晨反例。新例子仅用于声音控制试听，不参与训练或文本模型分数统计；没有冒充新LoRA预测。此前请求参数一致的原始片段可按指纹复用，旧音频和意见保留。

第五轮的实际试听时长、语速、响度和边界测量见[第五轮处理记录](eval/voxcpm_feedback_review_v5_20261009.md)。

最新意见认可语速，选择“更明显”，要求加强编号1、2的局部情绪，同时排查编号3、4连平静段落也用力的问题。第六轮使用[voxcpm_audition_v5.json](eval/voxcpm_audition_v5.json)，正文、标点与人物沿用，优先生成原编号1–4及一份纯平静对照（本轮编号5，沿用上一轮编号6正文）。上一轮编号5未修改，仍保存在旧运行目录。

已确认旧版1、2合并跨语气旁白为一次请求，仅使用整段“细微起伏”指令；旧版3、4的平静请求仍使用平静指令，没有统一变成紧张。旧处理器固定使用`mild`，此次选中的`strong`此前也未应用到旧音频。现在反馈处理器采用实际保存的强度，三个选项都有完整配置。`strong`在语气变化处拆分，迟疑与紧绷使用局部指令，对白使用单独的日常表达；平静、冷静共用平和指令，取消全局加强后缀。同人物同语气仍合并，句号边界目标350ms，逗号边界180ms；已认可的atempo=0.94及角色切换目标沿用。

旧版3的迟疑句曾获得约+16.08dB固定增益，4的冷静句约+5.47dB；其他平静句多数降低了音量。大增益可能突出气声与用力感，但音量测量不能证明全部过度演绎都由响度造成。第六轮平静/冷静目标降低1LU，低声/压低/迟疑降低2LU，正增益最多12dB，继续保留人物音量差及-2dBTP真峰值保护。3、4另附“旧合成，仅调整响度”对照，严格复用旧原始片段、指令、语速、裁切位置和静音帧；该对照不调用模型，帮助区分音量变化与韵律问题。核验还发现本轮4末句后存在近10秒全零尾部，现清除超长数字静音并保留80ms保护区；低幅度非零尾声仍受1200ms裁切上限保护，句内停顿保留。后处理修订可从最近完成轮次复用请求完全一致的原始片段，不必重新调用模型。原因、请求及测量见[第六轮处理记录](eval/voxcpm_feedback_review_v6_20261009.md)。

重新打开Notebook 07并按步骤1、2、3运行：先听五份本轮音频，旧版及只改响度对照折叠显示，可按需回听；只需记下有问题的编号和现象。三个意见字段都可修改，新表单沿用本轮实际强度，备注留空等待新评价，保存才提交；Notebook不合成、不训练、不启动服务。本轮仍在校准声音控制，没有重训文本LoRA。

`scripts/apply_voxcpm_feedback.py`读取明确保存的用户意见，执行agent已审阅的稿件与配置修订；不自动理解备注或学习新策略。它保留原意见与音频，生成新结果后才更新试听指针。同输入再次执行可复用结果；始终复用用户手动启动的服务，不自动启动API。
