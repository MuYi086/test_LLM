"""按已保存的强度执行审阅后的声音修订，保留旧版与只改响度的对照。"""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from check_environment import PROJECT_ROOT
from run_second_iteration import iteration_lock
from run_voxcpm_audition import ensure_reference, join_segments, local_service, save, wav_info
from voxcpm_adapter import compile_segments, fingerprint, load_delivery, parse_script, verify_review
from voxcpm_audio_levels import adjust_tempo, join_level_comparison, match_segment_level

LATEST = PROJECT_ROOT / "outputs/latest_voxcpm_audition.json"
REVISED_AUDITION = PROJECT_ROOT / "eval/voxcpm_audition_v5.json"


def find_feedback():
    """取用户最近一次明确保存的意见，不把预选项当作提交。"""
    paths = list((PROJECT_ROOT / "outputs").glob("voxcpm-*/user_decision.json"))
    if not paths:
        raise FileNotFoundError("尚未找到用户保存的试听意见")
    return max(paths, key=lambda path: json.loads(path.read_text())["created_at"])


def build_feedback_plan(bundle, previous, previous_root, delivery, strength=None):
    """按稿件顺序生成当前试听，另附选定的旧版本，不限制为固定两条故事。"""
    cases = {case["id"]: case for case in bundle["cases"]}
    plan = []
    previous_bundle = json.loads(Path(previous["bundle_file"]).read_text())
    previous_cases = {case["id"]: case for case in previous_bundle["cases"]}
    for number, identifier in enumerate(bundle["audition_order"], 1):
        case = cases[identifier]
        rows = parse_script(case["script"], case["allowed_roles"], delivery["tones"])
        verify_review(case, rows)
        entry = {
            "id": identifier,
            "title": case["title"],
            "variant": "tone_local_revision",
            "label": f"本轮 {number}：{case['title']}",
            "provenance": case["provenance"],
            "script": case["script"],
            "segments": [],
            "optional_comparison": False,
        }
        if identifier in bundle["previous_comparison_ids"]:
            candidates = [
                e
                for e in previous["entries"]
                if e["id"] == identifier
                and not e.get("optional_comparison", bool(e.get("reused_from")))
            ]
            if len(candidates) != 1:
                raise ValueError("缺少唯一的上一轮对照音频")
            prior = {
                **entry,
                "variant": "previous_reference",
                "label": "上一轮完整版本（可选对照）",
                "reuse_audio": str(previous_root / candidates[0]["audio_file"]),
                "script": candidates[0].get("script", previous_cases[identifier]["script"]),
                "provenance": candidates[0]["provenance"],
                "optional_comparison": True,
            }
            plan.append(prior)
        if case["audition_kind"] not in {"roles", "narration"}:
            raise ValueError("未知试听类型")
        voices = (
            bundle["provisional_voices"]
            if case["audition_kind"] == "roles"
            else dict.fromkeys(case["allowed_roles"], bundle["provisional_voices"]["旁白"])
        )
        entry["segments"] = compile_segments(rows, voices, delivery, strength)
        for segment in entry["segments"]:
            # 原始TTS指令、种子与参数沿用；放缓发生在合成后的atempo处理。
            segment["request"].update(
                cfg_value=2.0,
                inference_timesteps=10,
                normalize=False,
                denoise=False,
                retry_badcase=True,
                load_denoiser=False,
                optimize=False,
                device="cuda",
            )
        plan.append(entry)
        if identifier in bundle.get("level_only_comparison_ids", []):
            candidates = [
                e
                for e in previous["entries"]
                if e["id"] == identifier
                and not e.get("optional_comparison", bool(e.get("reused_from")))
            ]
            if len(candidates) != 1:
                raise ValueError("缺少只改响度对照的唯一原始版本")
            records = json.loads(
                (
                    previous_root / Path(candidates[0]["audio_file"]).parent / "requests.json"
                ).read_text()
            )
            keys = (
                "speaker",
                "tone",
                "source_tones",
                "pause_after_ms",
                "pause_mode",
                "transition_after",
                "request",
            )
            plan.append(
                {
                    **entry,
                    "variant": "level_only_comparison",
                    "label": "上一轮合成，仅调整响度（可选定位原因）",
                    "script": candidates[0].get("script", previous_cases[identifier]["script"]),
                    "segments": [{k: record[k] for k in keys} for record in records],
                    "optional_comparison": True,
                    "raw_cache_required": True,
                    "baseline_audio": candidates[0]["audio"],
                }
            )
    return plan


def previous_segment_cache(previous, previous_root, plan):
    """仅复用请求正文、指令、音色及全部参数一致的原始片段，声音处理另做。"""
    wanted = {fingerprint(s["request"]) for e in plan for s in e["segments"]}
    cache = {}
    for entry in previous["entries"]:
        folder = previous_root / Path(entry["audio_file"]).parent
        request_file = folder / "requests.json"
        if entry.get("reused_from") or not request_file.is_file():
            continue
        for index, record in enumerate(json.loads(request_file.read_text()), 1):
            key = fingerprint(record["request"])
            if key not in wanted:
                continue
            path = folder / f"segment-{index:02}.wav"
            wav_info(path.read_bytes())
            cache[key] = {
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return cache


def diagnose_previous(previous, root):
    """保存旧请求的客观证据，把程序原因与仍需试听的推测分开。"""
    inspected = []
    for entry in previous["entries"]:
        if entry.get("optional_comparison", bool(entry.get("reused_from"))):
            continue
        if entry["id"] not in {"sus_test_06", "sus_aud_v5_01", "sus_aud_v5_02", "sus_aud_v5_03"}:
            continue
        records = json.loads(
            (root / Path(entry["audio_file"]).parent / "requests.json").read_text()
        )
        inspected.append(
            {
                "id": entry["id"],
                "segments": [
                    {
                        "speaker": r["speaker"],
                        "source_tones": r["source_tones"],
                        "text": r["request"]["text"],
                        "instruction": r["request"]["control_instruction"],
                        "applied_gain_db": r.get("level_matching", {}).get("applied_gain_db"),
                    }
                    for r in records
                ],
            }
        )
    return {
        "previous_run": str(root),
        "inspected_requests": inspected,
        "confirmed": [
            "旧版1、2跨语气合并为一次请求，只收到细微起伏的整段指令，局部语气标签未分别控制声音。",
            "旧版3、4的平静请求仍使用平静指令，没有发现平静标签被统一改成紧张。",
            "旧版处理器固定使用mild；此次保存的strong是下一轮选择，不会改变已经生成的旧声音。",
            "旧版3的洛洵迟疑句固定增益约+16.08dB，4的洛洵冷静句约+5.47dB；其他若干平静句实际被降低音量。",
        ],
        "hypotheses": [
            "逐句匹配同一响度可能突出气声和用力感；固定增益只改变音量，不能把平静语气直接变成紧张。",
            "平静指令中的朗读措辞及参考声音的表达可能影响实际韵律；是否造成过度演绎需要对照试听，尚未唯一归因。",
        ],
    }


def execute(project, url, feedback_path, dry_run=False):
    """处理反馈，复用原音频和重复请求；不训练、不启动API、不修改用户意见。"""
    feedback_path = feedback_path.resolve()
    decision = json.loads(feedback_path.read_text(encoding="utf-8"))
    if decision["verdict"] != "revise":
        raise ValueError("本轮修订需要用户明确保存继续调整意见")
    previous_root = feedback_path.parent
    previous = json.loads((previous_root / "manifest.json").read_text())
    if previous["status"] != "complete":
        raise ValueError("反馈对应的试听未完成")
    bundle = json.loads(REVISED_AUDITION.read_text())
    delivery = load_delivery()
    strength = decision["strength"]
    if strength not in delivery.get("profiles", {}):
        raise ValueError("保存的演绎强度没有完整处理配置")
    profile = delivery["profiles"][strength]
    plan = build_feedback_plan(bundle, previous, previous_root, delivery, strength)
    cached_segments = previous_segment_cache(previous, previous_root, plan)
    if any(
        fingerprint(segment["request"]) not in cached_segments
        for entry in plan
        if entry.get("raw_cache_required")
        for segment in entry["segments"]
    ):
        raise ValueError("只改响度的对照缺少一致的原始片段，不能重新合成冒充对照")
    # 仅后处理或配置调整时，也复用最近已完成轮次中完全一致的原始请求。
    # 只改响度的旧片段优先保留，不能用新的合成替换它的基准。
    if LATEST.exists():
        latest_root = Path(json.loads(LATEST.read_text())["run_dir"])
        latest_manifest = json.loads((latest_root / "manifest.json").read_text())
        if latest_manifest["status"] == "complete" and latest_root != previous_root:
            cached_segments = {
                **previous_segment_cache(latest_manifest, latest_root, plan),
                **cached_segments,
            }
    references = {
        segment["request"]["audio_path"] for entry in plan for segment in entry["segments"]
    }
    state = {
        "feedback_sha256": hashlib.sha256(feedback_path.read_bytes()).hexdigest(),
        "delivery": delivery,
        "bundle": bundle,
        "plan": plan,
        # 缓存文件可以搬到新运行目录；同内容换路径不改变实验输入指纹。
        "cached_segments": {
            key: {"sha256": value["sha256"]} for key, value in cached_segments.items()
        },
        "references": {
            name: hashlib.sha256((project / "storage/timbre" / name).read_bytes()).hexdigest()
            for name in sorted(references)
        },
        "reused_audio": {
            e["reuse_audio"]: hashlib.sha256(Path(e["reuse_audio"]).read_bytes()).hexdigest()
            for e in plan
            if e.get("reuse_audio")
        },
        "scripts": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "voxcpm_adapter.py",
                "run_voxcpm_audition.py",
                "apply_voxcpm_feedback.py",
                "voxcpm_audio_levels.py",
            )
        },
        "ffmpeg_version": subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True, timeout=5
        ).stdout.splitlines()[0],
        "service": url,
        "model_config_sha256": hashlib.sha256(
            (Path.home() / "hf-mirror/openbmb/VoxCPM2/config.json").read_bytes()
        ).hexdigest(),
    }
    signature = fingerprint(state)
    if dry_run:
        print(
            f"[PASS] {len(bundle['cases'])}份本轮试听，另附{len(bundle['previous_comparison_ids'])}份旧版与{len(bundle.get('level_only_comparison_ids', []))}份只改响度对照。strength={strength}，tempo={profile['tempo']}，{len(cached_segments)}个旧片段可复用。反馈：{decision['comments']}"
        )
        return None
    if LATEST.exists():
        completed = Path(json.loads(LATEST.read_text())["run_dir"])
        metadata = json.loads((completed / "manifest.json").read_text())
        if (
            metadata["status"] == "complete"
            and metadata.get("state_sha256") == signature
            and all((completed / e["audio_file"]).is_file() for e in metadata["entries"])
        ):
            print(f"[COMPLETE] 已处理同一份意见，复用结果：{completed}")
            return completed
    root = (
        PROJECT_ROOT
        / "outputs"
        / ("voxcpm-feedback-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    root.mkdir()
    save(root / "delivery_snapshot.json", delivery)
    save(root / "source_feedback.json", decision)
    save(root / "bundle_snapshot.json", bundle)
    metadata = {
        "schema_version": 1,
        "status": "running",
        "revision": max(6, previous.get("revision", 1) + 1),
        "resolved_strength": strength,
        "state_sha256": signature,
        "state": state,
        "cached_segments": cached_segments,
        "entries": [],
        "bundle_file": str(root / "bundle_snapshot.json"),
        "source_bundle_file": str(REVISED_AUDITION),
        "feedback": {"source_file": str(feedback_path), "decision": decision},
        "diagnosis": diagnose_previous(previous, previous_root),
        "user_audio_decision": "pending",
        "scope": "保留已认可语速与角色切换间隔；按局部语气控制情绪，限制响度增益，另附只改响度对照；没有重训文本LoRA",
        "audition_summary": {
            "current_versions": len(bundle["cases"]),
            "new_examples": len(bundle["new_case_ids"]),
            "previous_comparisons": len(bundle["previous_comparison_ids"]),
            "level_only_comparisons": len(bundle.get("level_only_comparison_ids", [])),
            "tempo": profile["tempo"],
        },
    }
    save(root / "manifest.json", metadata)
    request_cache = {
        key: Path(value["file"]).read_bytes() for key, value in cached_segments.items()
    }
    cache_sources = {key: value["file"] for key, value in cached_segments.items()}
    new_requests = 0
    try:
        with local_service(project, url, root / "service.log") as health:
            metadata["service_health"] = health
            for name, sha256 in state["references"].items():
                ensure_reference(project, url, name, sha256)
            for index, entry in enumerate(plan, 1):
                print(f"[RUN] {index}/{len(plan)} {entry['label']}", flush=True)
                folder = root / (entry["id"] + "-" + entry["variant"])
                folder.mkdir()
                output = folder / "audio.wav"
                requests = []
                if entry.get("reuse_audio"):
                    output.write_bytes(Path(entry["reuse_audio"]).read_bytes())
                    info = wav_info(output.read_bytes())[0]
                else:
                    segment_paths = []
                    for segment_index, segment in enumerate(entry["segments"], 1):
                        payload = segment["request"]
                        key = fingerprint(payload)
                        cached = key in request_cache
                        cache_source = cache_sources.get(key) if cached else None
                        started = time.monotonic()
                        print(
                            f"[PROGRESS] {segment_index}/{len(entry['segments'])} {segment['speaker']} {'复用' if cached else '合成'}",
                            flush=True,
                        )
                        if cached:
                            data = request_cache[key]
                        else:
                            if entry.get("raw_cache_required"):
                                raise ValueError("只改响度的对照必须复用上一轮原始合成")
                            request = urllib.request.Request(
                                url + "/v1/voxcpm2/clone",
                                data=json.dumps(payload, ensure_ascii=False).encode(),
                                headers={"Content-Type": "application/json"},
                            )
                            try:
                                with urllib.request.urlopen(
                                    request, timeout=delivery["request_timeout_seconds"]
                                ) as response:
                                    data = response.read()
                            except urllib.error.HTTPError as error:
                                raise RuntimeError(
                                    f"VoxCPM2 HTTP {error.code}: {error.read().decode(errors='replace')}"
                                ) from error
                            wav_info(data)
                            request_cache[key] = data
                            cache_sources[key] = "current_run"
                            new_requests += 1
                        path = folder / f"segment-{segment_index:02}.wav"
                        path.write_bytes(data)
                        tempo_path = folder / f"tempo-{segment_index:02}.wav"
                        tempo_processing = adjust_tempo(path, tempo_path, profile["tempo"])
                        leveled_path = folder / f"leveled-{segment_index:02}.wav"
                        level_matching = match_segment_level(
                            tempo_path,
                            leveled_path,
                            segment["speaker"],
                            profile["loudness"],
                            tone=segment["source_tones"][0]
                            if len(segment["source_tones"]) == 1
                            else segment["tone"],
                        )
                        segment_paths.append(leveled_path)
                        requests.append(
                            {
                                **segment,
                                "cache_reused": cached,
                                "cache_source": cache_source,
                                "elapsed_seconds": round(time.monotonic() - started, 3),
                                "audio": wav_info(data)[0],
                                "tempo_processing": tempo_processing,
                                "level_matching": level_matching,
                            }
                        )
                    if entry.get("raw_cache_required"):
                        info = join_level_comparison(segment_paths, output, entry["baseline_audio"])
                    else:
                        info = join_segments(
                            segment_paths,
                            [s["pause_after_ms"] for s in entry["segments"]],
                            output,
                            profile["edge_trim"],
                            pause_mode=profile["pause_mode"],
                        )
                    info["level_matching"] = [r["level_matching"] for r in requests]
                    info["tempo_processing"] = [r["tempo_processing"] for r in requests]
                    save(folder / "requests.json", requests)
                record = {
                    key: entry[key]
                    for key in ("id", "title", "variant", "label", "provenance", "script")
                }
                record.update(
                    audio_file=str(output.relative_to(root)),
                    audio=info,
                    segment_count=len(entry["segments"]),
                    reused_from=entry.get("reuse_audio"),
                    optional_comparison=entry["optional_comparison"],
                    raw_cache_required=entry.get("raw_cache_required", False),
                )
                metadata["entries"].append(record)
                save(root / "manifest.json", metadata)
                print(f"[PASS] {entry['label']}：{info['duration_seconds']}秒", flush=True)
        metadata.update(
            status="complete",
            finished_at=datetime.now(UTC).isoformat(),
            unique_requests=new_requests,
            previous_cache_entries=len(cached_segments),
        )
        save(root / "manifest.json", metadata)
        (root / "feedback_review.md").write_text(feedback_report(metadata), encoding="utf-8")
        save(LATEST, {"run_dir": str(root)})
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(root / "manifest.json", metadata)
        raise
    print(f"[COMPLETE] 局部情绪修订与原因对照已完成：{root}", flush=True)
    return root


def feedback_report(metadata):
    """把实际生成及边界测量放在试听入口，避免把数值改善当作听感结论。"""
    rows = "\n".join(
        f"| {e['label']} | {e['audio']['duration_seconds']} | {'沿用' if e['reused_from'] else e['segment_count']} |"
        for e in metadata["entries"]
    )
    gaps = "\n".join(
        f"| {e['title']} / 第{b['after_segment']}段后 | {b['configured_ms']} | {b['existing_quiet_ms']} | {b['inserted_ms']} | {b['total_quiet_ms']} |"
        for e in metadata["entries"]
        if not e["reused_from"]
        for b in e["audio"].get("boundaries", [])
    )
    levels = "\n".join(
        f"| {e['title']} / {i} {level['speaker']} | {level['before']['integrated_lufs']} | {level['applied_gain_db']} | {level['after']['integrated_lufs']} | {level['after']['true_peak_dbtp']} |"
        for e in metadata["entries"]
        if not e["reused_from"]
        for i, level in enumerate(e["audio"].get("level_matching", []), 1)
    )
    confirmed = "\n".join("- " + item for item in metadata["diagnosis"]["confirmed"])
    hypotheses = "\n".join("- " + item for item in metadata["diagnosis"]["hypotheses"])
    return f"""# 第六轮：局部情绪与过度演绎排查

已处理你保存的意见：{metadata["feedback"]["decision"]["comments"]}

已采用保存的强度 `{metadata["resolved_strength"]}`。编号1、2仅在迟疑、紧绷处加强；编号3、4的对白使用克制的日常表达，平静、冷静共用平和指令。取消整段情绪覆盖和全局加强后缀，同人物同语气仍合并，语气变化处单独合成。

## 已确认的原因和证据

{confirmed}

本地接口 `voxcpm2/voxcpm2_helpers.py:58` 把指令组装为 `(instruction)正文`。它没有直接读取演绎稿的角色和情绪标签，一条指令作用于当前请求全文。旧请求及增益详情保存在manifest的diagnosis字段。

## 尚需试听判断的原因

{hypotheses}

3、4额外保留“上一轮合成，仅调整响度”：完全复用上一轮原始片段、语音指令、语速处理和边界停顿，只改变固定增益。它不调用模型重新合成。若该版就明显缓和，响度可能是主要因素；若仍用力而本轮改善，指令/本次合成差异可能更重要。该对照仍不能单独证明参考音色的因果作用。

## 本轮调整

1. 已认可的 atempo={metadata["audition_summary"]["tempo"]} 和旁白→对白400ms、对白→旁白500ms、角色→角色650ms沿用。新拆分的同人物句号边界目标350ms，逗号边界180ms；只补齐不足，低幅度非零尾声继续遵守1200ms裁切上限；额外清掉长段全零数字静音并保留80ms保护区，句内静音不裁。换指令可能改变模型原始语速，保持后处理系数不能保证所有句子的实际语速完全相同。
2. 平静、冷静目标降低1LU，低声、压低、迟疑降低2LU，人物音量差仍保留；正增益最多12dB，真峰值保护-2dBTP。保留句内动态，不用压缩器把所有表达拉到同样音量。受限片段如实记录，不声称所有片段都达到目标。
3. 优先听5份本轮音频：1、2情绪层次，3、4平静与对白是否收敛，5纯平静反例（沿用上一轮编号6的正文）。旧版与两份只改响度对照折叠显示，按需回听。上一轮编号5未修改，旧结果保留。

正文、标点、人物、对白及事实沿用并核对。本轮没有重新训练文本LoRA，独立试听素材不参与训练或模型分数统计。

| 版本 | 时长/秒 | 合成段数 |
| --- | --- | --- |
{rows}

实际新模型请求{metadata["unique_requests"]}次，{metadata["previous_cache_entries"]}个匹配的旧原始片段可直接复用。

| 片段/角色 | 输入/LUFS | 固定增益/dB | 处理后/LUFS | 真峰值/dBTP |
| --- | --- | --- | --- | --- |
{levels}

| 边界 | 目标/ms | 已有低幅度/ms | 补入/ms | 合计/ms |
| --- | --- | --- | --- | --- |
{gaps}

低幅度间隔以PCM幅度阈值32/32768测量，不等同于实际发声间隔或听感评分。边缘裁切保留80ms保护区，单边最多1200ms。WAV、响度和时间测量不能代替听感或漏读检查。

Notebook只查看结果，不启动服务或训练。步骤3三个字段都可修改，点击保存才提交意见；新表单沿用本轮实际采用的{metadata["resolved_strength"]}选择，备注留空等待你评价新声音，不把旧意见当作新提交。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument(
        "--service-project", type=Path, default=PROJECT_ROOT.parents[1] / "LLM-for-Local-Machine"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8322")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with iteration_lock() as acquired:
        if not acquired:
            print("[STATUS] 已有实验执行，本次不重复运行。")
            return
        execute(
            args.service_project.resolve(),
            args.base_url.rstrip("/"),
            args.feedback or find_feedback(),
            args.dry_run,
        )


if __name__ == "__main__":
    main()
