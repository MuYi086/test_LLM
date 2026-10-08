# 文本 SFT 数据

后续使用 ms-swift 的 `messages` JSONL 格式，每行是一条完整对话。训练集放 `train.jsonl`，验证集放 `val.jsonl`；真实数据默认忽略，不提交到 Git。

```json
{"messages":[{"role":"system","content":"你是一名有声剧导演。将小说改写为带角色与语气标记的演绎稿，不添加原文没有的情节。"},{"role":"user","content":"改写下面的文本：走廊尽头传来敲门声。林舟停下脚步，低声问：谁在那里？"},{"role":"assistant","content":"[旁白][低沉]走廊尽头传来敲门声。林舟停下脚步。\n[林舟][压低声音]谁在那里？"}]}
```

这条记录仅说明格式，不能用来评估微调收益。本阶段先验证环境；下一阶段准备教学数据，并固定训练前后的测试提示词。训练集和验证集应按来源文本拆分，避免同一段小说的近似改写落入两个集合。

参考：[ms-swift 自定义数据集](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)。
