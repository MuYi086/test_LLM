"""准备方案 A 的真实音频对照；复用一致结果，临时 API 服务结束后退出。"""

import argparse
import contextlib
import io
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import UTC, datetime
from pathlib import Path

from check_environment import PROJECT_ROOT
from prepare_voxcpm_audition import AUDITION
from run_second_iteration import iteration_lock
from voxcpm_adapter import compile_segments, fingerprint, load_delivery, parse_script, verify_review

LATEST = PROJECT_ROOT / "outputs/latest_voxcpm_audition.json"


def read_json_url(url, timeout=5):
    """读取本地接口 JSON，错误交由调用者处理。"""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def ensure_reference(project, url, name, sha256):
    """通过已有上传接口登记逻辑音色，复用共享音色文件而不覆盖旧引用。"""
    query = urllib.parse.urlencode({"file_name": name})
    check_url = url + "/v1/check/audio?" + query
    existing = read_json_url(check_url)
    if not existing.get("exists"):
        result = subprocess.run(
            [
                "curl",
                "--fail-with-body",
                "--silent",
                "--show-error",
                "--max-time",
                "30",
                url + "/v1/upload_audio",
                "--form",
                f"audio=@{project / 'storage/timbre' / name};type=audio/wav",
                "--form-string",
                f"full_path={name}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        json.loads(result.stdout)
        existing = read_json_url(check_url)
    if not existing.get("exists") or existing.get("sha256") != sha256:
        raise ValueError(f"参考音频逻辑映射与本地内容不一致：{name}")


@contextlib.contextmanager
def local_service(project, url, log_path, start_service=False):
    """复用已运行 API；仅关闭本次启动的 API，不停止用户已有服务。"""
    process = None
    log = None
    try:
        try:
            health = read_json_url(url + "/v1/health")
        except urllib.error.URLError:
            if not start_service:
                raise RuntimeError(
                    "VoxCPM2服务未启动。请手动执行bash start.sh；本程序默认不占用端口。"
                ) from None
            parsed = urllib.parse.urlsplit(url)
            if parsed.hostname != "127.0.0.1" or parsed.scheme != "http":
                raise ValueError("自动启动只支持本机 HTTP 接口") from None
            import os

            env = dict(os.environ, HOST="127.0.0.1", PORT=str(parsed.port or 8322))
            python = project / "voxcpm2/.venv/bin/python"
            if not python.is_file():
                raise FileNotFoundError(f"缺少现有 VoxCPM2 环境：{python}") from None
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [str(python), "-u", "main.py"],
                cwd=project / "voxcpm2",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"VoxCPM2 API 启动失败：{log_path}") from None
                try:
                    health = read_json_url(url + "/v1/health", timeout=1)
                    break
                except urllib.error.URLError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("VoxCPM2 API 启动超时") from None
                    time.sleep(0.25)
        if "voxcpm2_model_dir" not in health.get("paths", {}):
            # 当前本地 health 的 paths 字段用于确认端口属于预期服务。
            if "voxcpm2" not in json.dumps(health).lower():
                raise ValueError("端口不是 VoxCPM2 服务")
        yield health
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log is not None:
            log.close()


def build_plan(bundle, delivery, strength):
    """旁白对照固定同一音色；多角色示范单独列出，避免混淆对照变量。"""
    entries = []
    narrator = bundle["provisional_voices"]["旁白"]
    for case in bundle["cases"]:
        rows = parse_script(case["script"], case["allowed_roles"], delivery["tones"])
        verify_review(case, rows)
        if case["id"] in bundle["comparison_ids"]:
            voices = dict.fromkeys(case["allowed_roles"], narrator)
            variants = [
                (
                    "original",
                    "原文平读",
                    [{"speaker": "旁白", "tone": "平静", "text": case["source_text"]}],
                    False,
                ),
                ("rewrite_neutral", "改写后平读", rows, False),
                ("rewrite_controlled", "改写后可控演绎", rows, True),
            ]
        else:
            voices = bundle["provisional_voices"]
            variants = [("roles_controlled", "多角色演绎示范", rows, True)]
        for variant, label, content, controlled in variants:
            entries.append(
                {
                    "id": case["id"],
                    "title": case["title"],
                    "variant": variant,
                    "label": label,
                    "provenance": case["provenance"],
                    "segments": compile_segments(content, voices, delivery, strength, controlled),
                }
            )
    return entries


def wav_info(data):
    """验证响应是非空 PCM WAV，记录帧数、采样率与峰值。"""
    import array

    with wave.open(io.BytesIO(data), "rb") as audio:
        channels, width, rate, frames = (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getnframes(),
        )
        pcm = audio.readframes(frames)
    if channels != 1 or width != 2 or frames < rate // 4:
        raise ValueError("当前拼接器需要非空16位单声道PCM WAV")
    samples = array.array("h", pcm)
    if not samples or max(abs(x) for x in samples) == 0:
        raise ValueError("合成响应为静音")
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
        "frames": frames,
        "duration_seconds": round(frames / rate, 3),
        "peak": round(max(abs(x) for x in samples) / 32768, 5),
    }, pcm


def trim_edges(pcm, rate, policy):
    """仅裁掉PCM片段边缘的极低幅度样本，保留保护区及全部内部停顿。"""
    import array

    samples = array.array("h", pcm)
    threshold = policy["threshold_pcm"]
    if (
        not 0 <= threshold <= 64
        or not 0 <= policy["keep_ms"] <= 200
        or not 0 <= policy["max_trim_ms"] <= 1500
    ):
        raise ValueError("边缘静音配置越界")
    first = next((i for i, sample in enumerate(samples) if abs(sample) > threshold), None)
    if first is None:
        return pcm, {"head_frames": 0, "tail_frames": 0}
    tail = next(i for i, sample in enumerate(reversed(samples)) if abs(sample) > threshold)
    keep = int(rate * policy["keep_ms"] / 1000)
    cap = int(rate * policy["max_trim_ms"] / 1000)
    head_cut, tail_cut = min(max(first - keep, 0), cap), min(max(tail - keep, 0), cap)
    if policy.get("trim_exact_zeros", False):
        # 纯数字静音不含弱语音；仍为非零样本保留保护区，弱信号继续受原裁切上限保护。
        zero_head = next(i for i, sample in enumerate(samples) if sample != 0)
        zero_tail = next(i for i, sample in enumerate(reversed(samples)) if sample != 0)
        head_cut = max(head_cut, max(zero_head - keep, 0))
        tail_cut = max(tail_cut, max(zero_tail - keep, 0))
    return samples[head_cut : len(samples) - tail_cut].tobytes(), {
        "head_frames": head_cut,
        "tail_frames": tail_cut,
    }


def edge_quiet_frames(pcm, threshold):
    """测量两端连续低幅度区间；不把句内停顿当作拼接边界。"""
    import array

    samples = array.array("h", pcm)
    head = next((i for i, x in enumerate(samples) if abs(x) > threshold), len(samples))
    tail = next((i for i, x in enumerate(reversed(samples)) if abs(x) > threshold), len(samples))
    return head, tail


def join_segments(paths, pauses, output, edge_policy=None, pause_mode="additive"):
    """补齐角色边界的总低幅度间隔，或沿用固定追加静音；句内韵律保持原样。"""
    if not paths or len(paths) != len(pauses):
        raise ValueError("音频片段与停顿数量不一致")
    if pause_mode not in {"additive", "target_gap"}:
        raise ValueError("停顿模式无效")
    if pause_mode == "target_gap" and edge_policy is None:
        raise ValueError("总间隔模式需要低幅度阈值配置")
    segments = []
    trimmed = []
    expected = None
    for path in paths:
        info, pcm = wav_info(Path(path).read_bytes())
        shape = (info["channels"], info["sample_width"], info["sample_rate"])
        if expected is not None and expected != shape:
            raise ValueError("片段音频格式不一致")
        expected = shape
        if edge_policy:
            pcm, cuts = trim_edges(pcm, shape[2], edge_policy)
            trimmed.append(cuts)
        segments.append(pcm)
    pieces, boundaries = [], []
    quiet = (
        [edge_quiet_frames(pcm, edge_policy["threshold_pcm"]) for pcm in segments]
        if edge_policy
        else None
    )
    rate = expected[2]
    for index, pcm in enumerate(segments):
        pieces.append(pcm)
        if index + 1 < len(segments):
            if not 0 <= pauses[index] <= 2000:
                raise ValueError("停顿越界")
            target = int(rate * pauses[index] / 1000)
            existing = quiet[index][1] + quiet[index + 1][0] if quiet else 0
            added = max(target - existing, 0) if pause_mode == "target_gap" else target
            pieces.append(bytes(added * expected[0] * expected[1]))
            boundaries.append(
                {
                    "after_segment": index + 1,
                    "pause_mode": pause_mode,
                    "configured_ms": pauses[index],
                    "existing_quiet_ms": round(existing * 1000 / rate, 3) if quiet else None,
                    "inserted_ms": round(added * 1000 / rate, 3),
                    "total_quiet_ms": round((existing + added) * 1000 / rate, 3) if quiet else None,
                }
            )
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(expected[0])
        audio.setsampwidth(expected[1])
        audio.setframerate(expected[2])
        audio.writeframes(b"".join(pieces))
    info = wav_info(output.read_bytes())[0]
    if edge_policy:
        info["edge_trims"] = trimmed
    info["boundaries"] = boundaries
    return info


def save(path, value):
    """原子保存运行记录。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute(project, url, strength, dry_run=False, start_service=False):
    """串行生成7份试听，保留每次请求和失败原因；不启动文本训练。"""
    bundle = json.loads(AUDITION.read_text(encoding="utf-8"))
    delivery = load_delivery()
    strength = strength or delivery["default_strength"]
    plan = build_plan(bundle, delivery, strength)
    reference_files = sorted(
        {seg["request"]["audio_path"] for entry in plan for seg in entry["segments"]}
    )
    references = {}
    import hashlib

    for name in reference_files:
        path = project / "storage/timbre" / name
        references[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    state = {
        "plan": plan,
        "references": references,
        "strength": strength,
        "service": url,
        "model_config_sha256": hashlib.sha256(
            (Path.home() / "hf-mirror/openbmb/VoxCPM2/config.json").read_bytes()
        ).hexdigest(),
        "adapter_script_sha256": hashlib.sha256(
            Path(__file__).with_name("voxcpm_adapter.py").read_bytes()
        ).hexdigest(),
    }
    signature = fingerprint(state)
    if dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        print(
            f"[PASS] 计划核对：{len(plan)}份试听，{sum(len(e['segments']) for e in plan)}次合成；未加载权重。"
        )
        return None
    if LATEST.exists():
        old = Path(json.loads(LATEST.read_text())["run_dir"])
        previous = json.loads((old / "manifest.json").read_text())
        if (
            previous["status"] == "complete"
            and previous["state_sha256"] == signature
            and all((old / e["audio_file"]).is_file() for e in previous["entries"])
        ):
            print(f"[COMPLETE] 复用已完成的试听：{old}")
            return old
    root = (
        PROJECT_ROOT
        / "outputs"
        / ("voxcpm-audition-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    root.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "status": "running",
        "state_sha256": signature,
        "state": state,
        "entries": [],
        "user_audio_decision": "pending",
        "scope": "PCM结构检查通过不代表听感、音色或转写正确；需要用户试听",
    }
    save(root / "manifest.json", metadata)
    try:
        with local_service(project, url, root / "service.log", start_service) as health:
            metadata["service_health"] = health
            for name in reference_files:
                ensure_reference(project, url, name, references[name])
            for index, entry in enumerate(plan, 1):
                print(f"[RUN] {index}/{len(plan)} {entry['title']}：{entry['label']}", flush=True)
                folder = root / (entry["id"] + "-" + entry["variant"])
                folder.mkdir()
                segments, requests = [], []
                for segment_index, segment in enumerate(entry["segments"], 1):
                    payload = segment["request"]
                    print(
                        f"[PROGRESS] 片段 {segment_index}/{len(entry['segments'])} {segment['speaker']}/{segment['tone']}",
                        flush=True,
                    )
                    request = urllib.request.Request(
                        url + "/v1/voxcpm2/clone",
                        data=json.dumps(payload, ensure_ascii=False).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    started = time.monotonic()
                    try:
                        with urllib.request.urlopen(
                            request, timeout=delivery["request_timeout_seconds"]
                        ) as response:
                            data = response.read()
                    except urllib.error.HTTPError as error:
                        raise RuntimeError(
                            f"VoxCPM2 HTTP {error.code}: {error.read().decode(errors='replace')}"
                        ) from error
                    info, _ = wav_info(data)
                    target = folder / f"segment-{segment_index:02}.wav"
                    target.write_bytes(data)
                    segments.append(target)
                    requests.append(
                        {
                            **segment,
                            "audio": info,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                output = folder / "audio.wav"
                info = join_segments(
                    segments,
                    [s["pause_after_ms"] for s in entry["segments"]],
                    output,
                    edge_policy=delivery.get("profiles", {}).get(strength, {}).get("edge_trim")
                    if entry["variant"] in {"rewrite_controlled", "roles_controlled"}
                    else None,
                    pause_mode=delivery.get("profiles", {})
                    .get(strength, {})
                    .get("pause_mode", "additive")
                    if entry["variant"] in {"rewrite_controlled", "roles_controlled"}
                    else "additive",
                )
                save(folder / "requests.json", requests)
                metadata["entries"].append(
                    {key: entry[key] for key in ("id", "title", "variant", "label", "provenance")}
                    | {"audio_file": str(output.relative_to(root)), "audio": info}
                )
                save(root / "manifest.json", metadata)
                print(f"[PASS] {entry['label']}：{info['duration_seconds']}秒", flush=True)
        metadata.update(status="complete", finished_at=datetime.now(UTC).isoformat())
        save(root / "manifest.json", metadata)
        save(LATEST, {"run_dir": str(root)})
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(root / "manifest.json", metadata)
        raise
    print(f"[COMPLETE] 7份试听已生成：{root}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-project", type=Path, default=PROJECT_ROOT.parents[1] / "LLM-for-Local-Machine"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8322")
    parser.add_argument(
        "--strength",
        choices=["mild", "moderate", "strong"],
        help="默认读取控制词典的default_strength",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-service",
        action="store_true",
        help="显式允许临时启动API；完成后关闭。默认只复用手动启动的服务",
    )
    args = parser.parse_args()
    with iteration_lock() as acquired:
        if not acquired:
            print("[STATUS] 已有文本实验或试听执行，本次不重复启动。")
            return
        execute(
            args.service_project.resolve(),
            args.base_url.rstrip("/"),
            args.strength,
            args.dry_run,
            args.start_service,
        )


if __name__ == "__main__":
    main()
