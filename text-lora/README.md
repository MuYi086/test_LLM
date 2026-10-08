# 文本 LoRA：Qwen3.5-4B + ms-swift

按照[分享对话](https://chatgpt.com/share/6ac5c263-3e08-83ec-8240-d3df08cce17d)在当前仓库落地文本微调项目。第一阶段环境检查已通过，第二阶段的原模型基线推理脚本和 Notebook 已准备好。后续在这里维护数据处理、LoRA 训练与效果对比。

## 目录和当前进度

```text
text-lora/
├── .python-version                 # 3.12.13，与本仓库已有环境一致
├── pyproject.toml                  # uv 依赖声明
├── configs/                       # 本地模型路径、固定推理参数
├── notebooks/
│   ├── 01_environment.ipynb        # 项目 kernel / GPU / 依赖 / tokenizer 检查
│   └── 02_base_inference.ipynb     # 单条推理、10 条固定测试与原模型基线
├── scripts/
│   ├── check_environment.py        # 环境检查
│   └── run_inference.py            # 基线与未来 adapter 推理共用入口
├── eval/                          # 固定测试提示词与人工检查标准
├── datasets/README.md              # 后续 SFT 数据约定
└── outputs/                        # 后续 adapter、checkpoint、评估结果
```

`.venv/` 和 `uv.lock` 由首次手动安装依赖时生成。`uv.lock` 应提交到 Git，`.venv/`、真实数据和训练结果已忽略。当前环境已安装依赖，命令行的 Python、依赖导入、CUDA/BF16 和离线 tokenizer 检查全部通过；`text-lora` kernel 已注册。

本项目单独管理 uv 环境。JupyterLab 继续使用全局 `uv tool` 环境，Notebook 使用 `text-lora/.venv` 的 kernel。原有 `Qwen3.5-4B-vLLM/` 与 `Qwen3.5-4B-LoRA和SwanLab/` 属于其他实验目录，新的 ms-swift 工作流统一放这里。图片 LoRA 按分享对话的顺序留待后续建立独立项目。

## 已确认的本地模型

2026-10-07 检查了模型配置、tokenizer 文件和权重索引引用的分片；两套模型的文件均存在且非空，尚未执行权重加载或训练。

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

推理使用已安装的 ms-swift `TransformersEngine`，参数按[官方 Python 推理接口](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html#using-python)和本地 4.5.3 代码核对。只使用本地模型路径，启用 Hugging Face 离线模式；BF16、单条、SDPA 路线不新增依赖、不使用量化。当前已验证输入与 API 调用结构，真实权重加载和生成留给 Notebook 执行。

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

## 第三阶段：教学数据与 LoRA

完整基线保存后，准备独立的 20～50 条教学 SFT 数据及验证集，再添加训练入口。16GB 显卡第一轮从以下参数开始，属于待实测的建议：

```text
BF16 / LoRA rank=8 / alpha=32 / target_modules=all-linear
max_length=512 / batch_size=1 / gradient_accumulation_steps=8
gradient_checkpointing=true
```

文本数据使用 `messages` JSONL。训练时先释放其他 GPU 任务；Qwen3.5 是多模态架构，纯文本实验需明确冻结视觉部分。训练通过后逐步尝试 1024、2048 的长度，记录 loss、峰值显存和固定提示词的输出。备用 Qwen3-4B 的显存经验不能直接视为 Qwen3.5-4B 的实测值。
