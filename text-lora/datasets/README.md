# 文本 SFT 数据

使用 ms-swift 的 `messages` JSONL 格式，每行是一条完整对话。训练集放 `train.jsonl`，验证集放 `val.jsonl`；真实数据默认忽略，不提交到 Git。

原模型基线已保存并完成人工评审。[教学数据标注指南](annotation_guide.md)中的 5 条样本已由用户执行 Notebook 03、批准并导出；agent 已扩充到 30 条训练数据和 6 条独立验证数据。agent 负责数据编写、修订与技术检查，你主要负责监督和决策。

配套入口是 `../notebooks/03_dataset_preparation.ipynb`，5 个全新片段的答案均已由 agent 编写完整。[样本与检查摘要](pilot_review.md)供你审阅；格式、ms-swift 训练模板 token 检查及审阅决策通过后，导出至 `processed/pilot-时间戳/`。该目录只用于教学数据准备，Notebook 不启动训练。

完整可复现语料在 [teaching_samples.json](teaching_samples.json)，含 id、source_id、类别、split、检查项和目标答案。它是原创教学样本，随项目保存；导出的 JSONL 和运行输出仍被 Git 忽略。已批准的 5 条内容保持原样；新增样本记录 agent 核对，用户审阅状态保持待决策。[扩充核对记录](expanded_review.md)说明数量与检查边界。

`scripts/prepare_training_data.py --check-tokens` 自动生成 `train.jsonl`、`val.jsonl`，保存 `processed/teaching-v1/sources.jsonl` 和 `checks.json`。重复执行复用相同数据，拒绝覆盖不同内容。训练集和验证集按故事来源分离，现有 `eval/prompts.jsonl` 的固定输入与其改写不得加入任一集合。程序检查完全重复与文本相似度，不能替代来源和事实复核。

后续打开 `../notebooks/04_lora_training.ipynb`，按数据监督 → 试跑 → 训练与对照执行；默认复用本轮已完成的匹配结果。

读完 04 后，进入 `../notebooks/05_data_iteration.ipynb` 的第二轮实验。新增数据在 [teaching_samples_v2.json](teaching_samples_v2.json)，共 50 train / 12 val，独立导出至 `processed/teaching-v2/`。[第二轮核对记录](iteration2_review.md)说明动作与台词检查、新未见测试、用户监督和自动执行流程。原有 train/val 和第一轮快照继续保留。

参考：[ms-swift 4.5.3 自定义数据集](https://swift.readthedocs.io/en/v4.5/Customization/Custom-dataset.html)。
