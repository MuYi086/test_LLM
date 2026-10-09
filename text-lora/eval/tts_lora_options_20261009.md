# 悬疑小说 LoRA 与本地 TTS：两套待确认方案

2026-10-09。依据本地 `LLM-for-Local-Machine/README.md`、五个克隆服务的请求校验与 worker，以及模型官方文档。本文件为方案评审，不修改服务、不启动新训练、不生成音频；用户确认后再实施。

目标：让恐惧、悬疑小说更适合有氛围感的朗读，保留原文事实、人物、动作、数量、否定与未知信息。微调对象仍为文本模型，声音由现有 TTS 克隆服务生成。本次推荐依据接口能力与接入方式，没有把五个模型的实际音质做过试听排名。

## 已确认的接口能力

| 克隆接口 | 当前本地封装的表达控制 | 对训练目标的影响 |
| --- | --- | --- |
| VoxCPM2：8322 `/v1/voxcpm2/clone` | `clone_mode=controllable` + `control_instruction`；同时使用参考音频 | 能把演绎语气映射成自然语言控制指令，最适合逐段多角色与情绪切换 |
| Qwen3-TTS Base：8321 `/v1/qwen/clone` | 参考音频 + 可选准确 `prompt_text`；没有接入自然语言情绪字段 | 文本负责措辞与节奏，参考音频引导音色与表达；效果需试听 |
| LongCat：8323 `/v1/longCat/clone` | 参考音频 + 必须准确转写；`duration_scale` 调整生成时长估计 | 可作为参考驱动的替换引擎，不把时长参数当情绪控制 |
| dots：8324 `/v1/dotsTTS/clone` | continuation 或 x-vector-only 克隆；本地未接情绪指令字段 | 可用于相同纯文本方案的对照，情绪不能依赖方括号标签 |
| FireRedTTS3 Base：8325 `/v1/FireRedTTS3/clone` | 参考音频 + 必须准确转写；Instruct 音色设计在另一个模式/端口 | Base 克隆不能套用 8304 Instruct 的指令式音色设计能力 |

这五个接口都拒绝 `style_prompt`。`[人物][情绪]` 是我们自己的中间表示，不是通用 TTS 控制协议；不能直接把它交给五个接口并假定会改变声音。

VoxCPM2 的可控模式不允许传 `prompt_text`，也忽略已上传的参考转写 sidecar；普通/极致克隆模式与控制指令互斥。`nonverbal_tags` 只允许最多一个白名单项，它们不等于恐惧、紧张等情绪分类。无原文依据时不增加笑声、叹息等声音。

## 方案 A：悬疑演绎稿 LoRA + VoxCPM2 可控克隆（优先推荐）

### 训练目标

LoRA 学习适度悬疑重写、完整事实保留、正确旁白/对白归属，以及保守的演绎标记。沿用可解析的 `[角色][语气]正文`，把它当中间稿。情绪与发声方式在实现中分别整理，避免把“压低”当人物心理事实，或把每个疑问都演成恐惧。

示意输入：`灯灭了，但门外的脚步声还在。`

```text
[旁白][压低]灯灭了。
[旁白][紧绷]门外的脚步声，却还在。
```

### 合成方式

程序解析标签，人物映射到既有参考音色，语气映射到经过试听的控制词典；只把正文放入 HTTP 请求的 `text`。例如“紧绷”先尝试克制紧张、略慢、吐字清晰，不指示模型添加喘息、尖叫或新的情节。

以下为请求结构示例，参考音频逻辑标识只是示意，没有实际调用：

```json
{
  "text": "门外的脚步声，却还在。",
  "audio_path": "narrator_reference.wav",
  "clone_mode": "controllable",
  "control_instruction": "用克制而紧张的语气朗读，音量压低，语速稍慢，吐字清晰。"
}
```

角色参考映射、服务地址、请求字段和停顿数值由程序配置，训练答案不硬编码本机路径或 HTTP JSON。相邻同角色、同语气的短行合并合成，以减少一次性 worker 的重复加载。跨角色或语气切换由编排层分段。

优势是角色与情绪能实际落到当前接口，最贴近恐惧、悬疑演绎目标；代价是多段合成与拼接更复杂，过短片段可能降低连贯性。VoxCPM2 官方也说明可控生成的一致性需要试听验证，不能把自然语言指令当作确定的声音结果。

## 方案 B：悬疑纯文本润色 LoRA + Qwen3-TTS Base 参考驱动朗读

### 训练目标

LoRA 输出干净的小说正文，改善长句、断句、措辞和线索强调，不把旁白转成剧本台词，不输出人物/情绪标签。保留“某人说/问”等必要归属，使同一旁白音色朗读时仍能理解谁说了什么。

同一示意输入的目标：

```text
灯灭了。门外的脚步声，却还在。
```

本地后端已经有“小说片段 → 纯净可朗读文本”的实验约定，可参考其数据思路；它使用 Qwen3-4B-Instruct / LLaMA-Factory，而当前项目为 Qwen3.5-4B / ms-swift，不能直接把旧配置或 adapter 混用。

### 合成方式

先以 Qwen3-TTS Base 的参考音频与准确转写做旁白朗读；需要更紧张的表达时，从同一说话者的克制、低声、紧张参考中选择。参考音频是否能稳定传递期望的表达是待试听的假设，不保证精确情绪强度。优先使用带准确转写的参考条件；仅音色向量模式不提供参考语音内容和韵律的同等条件。

其余三个参考克隆接口也可读同一份正文做对照：LongCat 与 FireRed Base 必须有准确参考转写，dots 可使用带转写 continuation。VoxCPM2 也可以普通/极致克隆方式参加这套方案。`prompt_text` 永远是参考音频的真实转写，不是目标稿，也不是情绪提示。

优势是同一份改写正文容易在五个模型间复用、适合旁白为主的长篇小说；代价是人物分别配音与逐句情绪切换较弱，声音氛围更依赖参考素材、标点及合成模型自身的表达能力。这里选择 Qwen 作为起点是因为现有参考条件与封装清晰，并非已经证明它的听感优于其余模型。

## 两套方案共用的验收方式

用户确认路线后，agent 先整理少量同输入、同人物音色的试听，再确定训练答案和控制词典。对照原文直接读、仅重写正文、重写并加可用的表达控制，分别判断措辞与声音控制的贡献。

内容检查覆盖动作、位置、数量、说话者、否定与未知身份；试听检查覆盖紧张感、自然度、角色音色连续性和长段衔接。平稳段落必须保持适度，不能全篇低声或紧绷。对需要修订的现有风格候选，不因标签齐全就视为合格稿。

`pause_ms` 在当前 Qwen/Vox/LongCat/dots worker 中用于内部块之间的静音，不是任意句内精确停顿，也不会自动替单条请求加末尾静音；逐句编排的停顿应在拼接层处理。FireRed Base 当前请求未提供这组分块/停顿参数。

agent 负责数据重写、事实检查、参考素材整理、适配程序、训练和对照；用户负责试听与决定风格强度、路线和是否采用。现有音色素材优先复用，再按缺口决定是否需要补充。

## 建议与待确认项

建议选 **A**：你的目标包含恐惧、悬疑的演绎氛围，现有 VoxCPM2 可控接口能够把语气真正接到声音控制。若优先追求一份正文在全部五个模型上复用、以旁白叙述为主，则选 **B**。

请用户确认 A 或 B；确认前保持当前数据、训练和服务实现不变。

## 核对来源

- [本地接口和参考转写约定](/home/muyi086/github/LLM-for-Local-Machine/README.md:660)
- [本地 VoxCPM2 请求约束](/home/muyi086/github/LLM-for-Local-Machine/voxcpm2/main.py:325)
- [本地 VoxCPM2 控制指令组装](/home/muyi086/github/LLM-for-Local-Machine/voxcpm2/voxcpm2_helpers.py:70)
- [本地 Qwen Base 请求字段](/home/muyi086/github/LLM-for-Local-Machine/qwen3_tts/main.py:339)
- [本地纯文本润色目标](/home/muyi086/github/LLM-for-Local-Machine/scripts/training/build_audio_polisher_poc.py:21)
- [VoxCPM2 官方可控克隆与限制](https://github.com/OpenBMB/VoxCPM)
- [Qwen3-TTS 官方 Base 克隆接口](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
- [LongCat 官方项目](https://github.com/meituan-longcat/LongCat-AudioDiT)
- [dots 官方项目](https://github.com/studio-dots-ai/dots.tts)
- [FireRedTTS3 官方 Base 与 Instruct 能力区分](https://github.com/FireRedTeam/FireRedTTS3)
