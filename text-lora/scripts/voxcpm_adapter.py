"""把已审阅的演绎稿映射到本地 VoxCPM2 可控克隆请求。"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

LINE = re.compile(r"^\[([^\[\]\n]+)\]\[([^\[\]\n]+)\](\S.*)$")
ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    """记录稿件、配置与请求的内容指纹，防止误复用旧样音。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_delivery(path=ROOT / "configs/voxcpm_delivery.json"):
    """读取语气词典；控制词仍需用户试听，不能视作声音效果保证。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema_version"] != 1 or data["max_segment_chars"] <= 0:
        raise ValueError("演绎配置无效")
    if data["default_strength"] not in data["strength_suffix"]:
        raise ValueError("默认强度不存在")
    for tone, spec in data["tones"].items():
        if not tone or not spec["instruction"].strip() or not 0 <= spec["pause_ms"] <= 2000:
            raise ValueError("演绎语气配置无效")
    for strength, profile in data.get("profiles", {}).items():
        if strength not in data["strength_suffix"] or not 0 <= profile.get("pause_ms", 0) <= 2000:
            raise ValueError("强度配置无效")
        if (
            set(profile.get("instructions", {})) | set(profile.get("dialogue_instructions", {}))
        ) - set(data["tones"]):
            raise ValueError("强度配置含未知语气")
        if profile.get("narrator_tone", "平静") not in data["tones"]:
            raise ValueError("旁白默认语气无效")
        if profile.get("pause_mode", "additive") not in {"additive", "target_gap"}:
            raise ValueError("停顿模式无效")
        if not 0.85 <= profile.get("tempo", 1) <= 1:
            raise ValueError("本轮仅支持轻微放缓语速")
        transitions = profile.get("transition_pause_ms", {})
        if set(transitions) - {
            "same_speaker",
            "narrator_to_dialogue",
            "dialogue_to_narrator",
            "dialogue_change",
        } or any(not 0 <= value <= 2000 for value in transitions.values()):
            raise ValueError("角色切换停顿配置无效")
        punctuation = profile.get("punctuation_pause_ms", {})
        if set(punctuation) - {"clause_end", "sentence_end"} or any(
            not 0 <= value <= 2000 for value in punctuation.values()
        ):
            raise ValueError("同人物标点停顿配置无效")
    return data


def parse_script(script, roles, tones):
    """拒绝未知标签和模型思考内容，避免把控制标记读入声音。"""
    rows = []
    for raw in script.splitlines():
        if not raw.strip():
            continue
        match = LINE.fullmatch(raw.strip())
        if not match:
            raise ValueError(f"无效演绎行：{raw}")
        speaker, tone, text = match.groups()
        if speaker not in roles or tone not in tones:
            raise ValueError(f"未知人物或语气：{speaker}/{tone}")
        if any(marker in text for marker in ("[", "]", "<think", "</think", "```")):
            raise ValueError("正文夹带控制标记或思考块")
        rows.append({"speaker": speaker, "tone": tone, "text": text})
    if not rows:
        raise ValueError("演绎稿为空")
    return rows


def verify_review(case, rows):
    """核对稿件审阅指纹、声明的事实及对白；不声称证明全部语义。"""
    reviewed = {key: case[key] for key in ("source_text", "script", "allowed_roles")}
    if case.get("review_status") != "agent_checked" or case.get("review_sha256") != fingerprint(
        reviewed
    ):
        raise ValueError("需要与当前原文及演绎稿一致的 agent 审阅记录")
    if "original_script" in case:
        # 标点修订必须保留全部正文字符及其顺序；对白另按人物、文字和次数核对。
        original = parse_script(
            case["original_script"], case["allowed_roles"], load_delivery()["tones"]
        )

        def content(lines):
            return "".join(
                c
                for row in lines
                for c in row["text"]
                if not c.isspace() and not unicodedata.category(c).startswith("P")
            )

        if content(original) != content(rows):
            raise ValueError("标点修订改变了正文内容或顺序")
    for fact in case["fact_checks"]:
        if not fact["source"] or fact["source"] not in case["source_text"]:
            raise ValueError("事实检查没有原文来源")
        text = "\n".join(r["text"] for r in rows if fact.get("role") in (None, r["speaker"]))
        if not fact["target"] or fact["target"] not in text:
            raise ValueError(f"声明的事实缺失或角色错误：{fact['target']}")
    for speech in case["speech_checks"]:
        actual = sum(
            r["speaker"] == speech["speaker"] and r["text"] == speech["text"] for r in rows
        )
        total = sum(r["text"].count(speech["text"]) for r in rows)
        if (
            actual != speech["count"]
            or total != speech["count"]
            or case["source_text"].count(speech["text"]) != speech["count"]
        ):
            raise ValueError(f"直接对白归属或次数错误：{speech['text']}")


def compile_segments(rows, voices, delivery, strength=None, controlled=True):
    """合并相邻同人物同语气的稿件，保留边界停顿并生成合法请求。"""
    strength = strength or delivery["default_strength"]
    if strength not in delivery["strength_suffix"]:
        raise ValueError("未知演绎强度")
    profile = delivery.get("profiles", {}).get(strength, {}) if controlled else {}
    merged = []
    limit = delivery["max_segment_chars"]
    for row in rows:
        if row["speaker"] not in voices or not voices[row["speaker"]]:
            raise ValueError(f"人物没有参考音色：{row['speaker']}")
        if len(row["text"]) > limit:
            raise ValueError("单行超过合成长度限制，需要先按句拆分和审阅")
        tone_key = row["tone"] if controlled else "平静"
        key = (row["speaker"], None if profile.get("merge_adjacent_speaker") else tone_key)
        if (
            merged
            and merged[-1]["key"] == key
            and len(merged[-1]["text"]) + len(row["text"]) <= limit
        ):
            merged[-1]["text"] += row["text"]
            merged[-1]["tone_weights"].append((tone_key, len(row["text"])))
        else:
            merged.append(
                {"key": key, "text": row["text"], "tone_weights": [(tone_key, len(row["text"]))]}
            )
    result = []
    for group in merged:
        speaker, tone = group["key"]
        source_tones = list(dict.fromkeys(t for t, _ in group["tone_weights"]))
        if tone is None:
            weights = {
                t: sum(w for label, w in group["tone_weights"] if label == t) for t in source_tones
            }
            tone = max(weights, key=weights.get)
        # 轻演绎不因一句紧绷而给整段旁白施加紧张指令；原标签留在来源记录中。
        if speaker == "旁白" and profile.get("narrator_tone"):
            tone = profile["narrator_tone"]
        spec = delivery["tones"][tone]
        instruction = spec["instruction"] if controlled else delivery["neutral_instruction"]
        instruction = profile.get("instructions", {}).get(tone, instruction)
        if speaker != "旁白":
            instruction = profile.get("dialogue_instructions", {}).get(tone, instruction)
        if (
            speaker == "旁白"
            and profile.get("narrator_variation_instruction")
            and set(source_tones) - {"平静", "冷静"}
        ):
            # 整段一次生成轻微起伏，保留句间韵律；平稳稿件仍使用原平静指令。
            instruction = profile["narrator_variation_instruction"]
        if (
            controlled
            and tone not in {"平静", "冷静"}
            and profile.get("append_strength_suffix", True)
        ):
            instruction += delivery["strength_suffix"][strength]
        result.append(
            {
                "speaker": speaker,
                "tone": tone,
                "source_tones": source_tones,
                "pause_after_ms": profile.get("pause_ms", spec["pause_ms"]),
                "pause_mode": profile.get("pause_mode", "additive"),
                "request": {
                    "text": group["text"],
                    "audio_path": voices[speaker],
                    "clone_mode": "controllable",
                    "control_instruction": instruction,
                    "seed": delivery["seed"],
                    "max_chars_per_chunk": 0,
                },
            }
        )
    for index, segment in enumerate(result):
        if index + 1 == len(result):
            transition = "end"
        else:
            speaker, next_speaker = segment["speaker"], result[index + 1]["speaker"]
            if speaker == next_speaker:
                transition = "same_speaker"
            elif speaker == "旁白":
                transition = "narrator_to_dialogue"
            elif next_speaker == "旁白":
                transition = "dialogue_to_narrator"
            else:
                transition = "dialogue_change"
        segment["transition_after"] = transition
        if transition in profile.get("transition_pause_ms", {}):
            segment["pause_after_ms"] = profile["transition_pause_ms"][transition]
        if transition == "same_speaker" and profile.get("punctuation_pause_ms"):
            text = segment["request"]["text"].rstrip("”’\"'）)")
            punctuation = "sentence_end" if text.endswith(tuple("。！？.!?")) else "clause_end"
            segment["pause_after_ms"] = profile["punctuation_pause_ms"][punctuation]
        if transition == "end" and segment["pause_mode"] == "target_gap":
            segment["pause_after_ms"] = 0
    return result
