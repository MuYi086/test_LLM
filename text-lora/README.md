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
│   └── 05_data_iteration.ipynb     # 第二轮 50/12 数据与原模型/两轮 LoRA 对照
├── scripts/
│   ├── check_environment.py        # 环境检查
│   ├── run_inference.py            # 基线与 adapter 推理共用入口
│   ├── prepare_training_data.py    # 原创语料导出、来源/格式/模板检查
│   ├── train_lora.py               # smoke/full 训练与 checkpoint 核验
│   ├── compare_inference.py        # 配置核对、按 ID 生成输出对照
│   ├── compare_iterations.py       # 原模型、第一轮、第二轮三组对照
│   └── run_second_iteration.py     # 第二轮一键检查、训练与评估
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

[第二轮方案与核对记录](datasets/iteration2_review.md)已落实为 50 条训练、12 条验证，完整模板最长 235 tokens；另写并固定 10 条未见测试及参考写法。数据独立保存在 teaching-v2，第一轮数据、Notebook、checkpoint 与复用记录保留。新增动作/台词约束、来源隔离、模板与训练参数预检查已通过；第二轮 GPU 训练尚未执行，效果须待实际输出评审。

入口 `scripts/run_second_iteration.py --dry-run` 可只检查计划；执行时自动完成试跑、正式训练、原测试第二轮生成、新测试三组生成与报告。按验证 loss 选模型，测试参考不进入模型输入或训练。成功记录到 `outputs/latest_iteration2.json`，再次运行复用匹配的完整结果。
