"""用本机FFmpeg测量响度，再以固定增益匹配短句，保留句内动态和全部时间位置。"""

import array
import json
import math
import subprocess
import wave
from pathlib import Path

from run_voxcpm_audition import edge_quiet_frames, wav_info


def adjust_tempo(source, output, factor):
    """以atempo轻微放缓片段，保留采样率与声线；边界停顿随后独立补齐。"""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("不能覆盖原始合成片段")
    if not 0.85 <= factor <= 1:
        raise ValueError("本轮仅支持轻微放缓语速")
    before = wav_info(source.read_bytes())[0]
    if factor == 1:
        output.write_bytes(source.read_bytes())
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(source),
                "-af",
                f"atempo={factor}",
                "-ar",
                str(before["sample_rate"]),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    after = wav_info(output.read_bytes())[0]
    if after["sample_rate"] != before["sample_rate"]:
        raise ValueError("语速处理不能修改采样率")
    return {"method": "ffmpeg_atempo", "factor": factor, "before": before, "after": after}


def measure_loudness(path):
    """读取loudnorm的输入LUFS与真峰值；测量输出不用于重采样或动态压缩。"""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-23:TP=-2:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(result.stderr[result.stderr.rfind("{") :])
    values = {"integrated_lufs": float(data["input_i"]), "true_peak_dbtp": float(data["input_tp"])}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("片段响度无法测量，不能静默套用增益")
    return values


def gain_plan(measured, target_lufs, policy):
    """匹配响度时为真峰值留余量，超出增益上限的片段明确记录限制。"""
    if not -40 <= target_lufs <= -12 or not -12 <= policy["true_peak_dbtp"] <= -1:
        raise ValueError("响度或峰值目标越界")
    if not 0 <= policy["max_gain_db"] <= 24 or not 0 <= policy["max_attenuation_db"] <= 24:
        raise ValueError("响度增益范围越界")
    requested = target_lufs - measured["integrated_lufs"]
    bounded = max(-policy["max_attenuation_db"], min(requested, policy["max_gain_db"]))
    safe_gain = policy["true_peak_dbtp"] - measured["true_peak_dbtp"]
    applied = min(bounded, safe_gain)
    return {
        "target_lufs": target_lufs,
        "requested_gain_db": round(requested, 3),
        "applied_gain_db": round(applied, 3),
        "limited_by_gain_range": not math.isclose(requested, bounded),
        "limited_by_true_peak": applied < bounded,
    }


def scale_pcm(pcm, gain_db):
    """整段施加相同增益，保持静音与样本数量；超出PCM范围时拒绝削波。"""
    gain = 10 ** (gain_db / 20)
    samples = [round(sample * gain) for sample in array.array("h", pcm)]
    if any(sample < -32768 or sample > 32767 for sample in samples):
        raise ValueError("固定增益会削波，需要先降低目标")
    return array.array("h", samples).tobytes()


def match_segment_level(source, output, speaker, policy, tone=None):
    """保留原始片段，另存响度匹配版本和测量，不改变语速、音高或句内停顿。"""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("不能覆盖原始合成片段")
    before = measure_loudness(source)
    target = (
        policy["target_lufs"]
        + policy.get("speaker_offsets_lu", {}).get(speaker, 0)
        + policy.get("tone_offsets_lu", {}).get(tone, 0)
    )
    plan = gain_plan(before, target, policy)
    info, pcm = wav_info(source.read_bytes())
    adjusted = scale_pcm(pcm, plan["applied_gain_db"])
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(info["channels"])
        audio.setsampwidth(info["sample_width"])
        audio.setframerate(info["sample_rate"])
        audio.writeframes(adjusted)
    after = measure_loudness(output)
    if after["true_peak_dbtp"] > policy["true_peak_dbtp"] + 0.05:
        raise ValueError("响度匹配后真峰值超出保护目标")
    return {"speaker": speaker, "tone": tone, "before": before, **plan, "after": after}


def join_level_comparison(paths, output, baseline, threshold=32):
    """只改响度对照沿用旧裁切和静音帧，避免音量改变影响阈值裁切位置。"""
    if len(paths) != len(baseline["edge_trims"]) or len(paths) - 1 != len(baseline["boundaries"]):
        raise ValueError("响度对照与旧版片段数量不符")
    rate = baseline["sample_rate"]
    segments = []
    for index, path in enumerate(paths):
        info, pcm = wav_info(Path(path).read_bytes())
        if (info["sample_rate"], info["channels"], info["sample_width"]) != (rate, 1, 2):
            raise ValueError("响度对照与旧版音频格式不符")
        if info["frames"] != baseline["tempo_processing"][index]["after"]["frames"]:
            raise ValueError("响度对照不能改变旧版片段时间位置")
        cuts = baseline["edge_trims"][index]
        start, stop = cuts["head_frames"], info["frames"] - cuts["tail_frames"]
        if not 0 <= start < stop <= info["frames"]:
            raise ValueError("旧版裁切记录无效")
        segments.append(pcm[start * 2 : stop * 2])
    pieces, boundaries = [], []
    quiet = [edge_quiet_frames(pcm, threshold) for pcm in segments]
    for index, pcm in enumerate(segments):
        pieces.append(pcm)
        if index + 1 < len(segments):
            prior = baseline["boundaries"][index]
            added = round(prior["inserted_ms"] * rate / 1000)
            pieces.append(bytes(added * 2))
            existing = quiet[index][1] + quiet[index + 1][0]
            boundaries.append(
                {
                    "after_segment": index + 1,
                    "pause_mode": "fixed_reference",
                    "configured_ms": prior["configured_ms"],
                    "existing_quiet_ms": round(existing * 1000 / rate, 3),
                    "inserted_ms": prior["inserted_ms"],
                    "total_quiet_ms": round((existing + added) * 1000 / rate, 3),
                }
            )
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"".join(pieces))
    info = wav_info(Path(output).read_bytes())[0]
    if info["frames"] != baseline["frames"]:
        raise ValueError("只改响度对照的总帧数必须等于旧版")
    return {
        **info,
        "edge_trims": baseline["edge_trims"],
        "boundaries": boundaries,
        "timing_method": "baseline_exact_cuts_and_padding",
    }
