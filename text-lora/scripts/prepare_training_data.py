"""导出原创教学语料并核对格式、来源隔离和真实训练模板长度。"""

import argparse
import copy
import hashlib
import json
import os
import re
import tomllib
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from check_environment import PROJECT_ROOT, load_model_config
from run_inference import load_prompts, prompts_fingerprint

CORPUS = PROJECT_ROOT / "datasets/teaching_samples.json"
TRAINING_CONFIG = PROJECT_ROOT / "configs/training.toml"
ROUND2_PROMPTS = PROJECT_ROOT / "eval/round2_prompts.jsonl"
ROUND2_BENCHMARK = PROJECT_ROOT / "eval/round2_benchmark.json"
TAG = re.compile(r"^\[([^\[\]\n]+)\]\[([^\[\]\n]+)\](\S.*)$")
EXPECTED = {
    "train": {
        "actions": 6,
        "three_roles": 5,
        "inner_thought": 5,
        "no_dialogue": 5,
        "ambiguity": 5,
        "general": 4,
    },
    "val": dict.fromkeys(
        ["actions", "three_roles", "inner_thought", "no_dialogue", "ambiguity", "general"], 1
    ),
}


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_training_config():
    with TRAINING_CONFIG.open("rb") as file:
        return tomllib.load(file)["training"]


def benchmark_prompts():
    if not ROUND2_BENCHMARK.exists():
        return []
    benchmark = json.loads(ROUND2_BENCHMARK.read_text(encoding="utf-8"))
    return [
        {key: case[key] for key in ("id", "source_id", "category", "messages", "review_criteria")}
        for case in benchmark["cases"]
    ]


def export_benchmark():
    """推理文件只含输入与检查项；参考答案留在独立 JSON 中。"""
    prompts = benchmark_prompts()
    if not prompts:
        raise ValueError("缺少第二轮未见评估集")
    cases = json.loads(ROUND2_BENCHMARK.read_text(encoding="utf-8"))["cases"]
    ids, sources = [c["id"] for c in cases], [c["source_id"] for c in cases]
    if len(cases) != 10 or len(set(ids)) != 10 or len(set(sources)) != 10:
        raise ValueError("第二轮评估需要 10 个独立案例")
    if Counter(case["category"] for case in cases) != {"tts_rewrite": 8, "general_control": 2}:
        raise ValueError("第二轮评估需要 8 条演绎任务与 2 条普通任务")
    for case in cases:
        if case["category"] == "tts_rewrite":
            reference = {
                **case,
                "messages": [
                    *case["messages"],
                    {"role": "assistant", "content": case["reference_output"]},
                ],
            }
            for line in case["reference_output"].splitlines():
                match = TAG.fullmatch(line)
                if (
                    not match
                    or match.group(1) not in case["allowed_roles"]
                    or match.group(3).rstrip("。！？?!，,；;：:") not in case["source_text"]
                ):
                    raise ValueError(f"{case['id']}: 参考答案无效")
            validate_content_checks(reference)
    content = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in prompts)
    if ROUND2_PROMPTS.exists() and ROUND2_PROMPTS.read_text(encoding="utf-8") != content:
        raise ValueError("拒绝覆盖已导出的第二轮固定测试")
    if not ROUND2_PROMPTS.exists():
        ROUND2_PROMPTS.write_text(content, encoding="utf-8")
    if load_prompts(ROUND2_PROMPTS) != prompts:
        raise ValueError("评估输入导出内容不一致")
    return prompts


def dataset_paths(corpus_path=CORPUS):
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    version = corpus.get("dataset_version", "teaching-v1")
    if not re.fullmatch(r"teaching-v[1-9][0-9]*", version):
        raise ValueError("数据版本必须采用 teaching-vN 格式")
    directory = PROJECT_ROOT / "datasets/processed" / version
    data_dir = PROJECT_ROOT / "datasets" if version == "teaching-v1" else directory
    return {
        "train": data_dir / "train.jsonl",
        "val": data_dir / "val.jsonl",
        "sources": directory / "sources.jsonl",
        "checks": directory / "checks.json",
    }


def validate_content_checks(sample):
    """检查人工声明的动作与台词约束；完整语义仍需逐句复核。"""
    identifier = sample["id"]
    lines = [TAG.fullmatch(line) for line in sample["messages"][-1]["content"].splitlines()]
    narrator = "\n".join(match.group(3) for match in lines if match and match.group(1) == "旁白")
    answer = sample["messages"][-1]["content"]
    for action in sample.get("narration_checks", []):
        if action not in sample["source_text"] or action not in narrator:
            raise ValueError(f"{identifier}: 声明的叙述事实缺失或未归旁白：{action}")
    for speech in sample.get("speech_checks", []):
        text, speaker, count = speech["text"], speech["speaker"], speech["count"]
        actual = sum(
            bool(match and match.group(1) == speaker and match.group(3) == text) for match in lines
        )
        if (
            actual != count
            or answer.count(text) != count
            or sample["source_text"].count(text) != count
        ):
            raise ValueError(f"{identifier}: 台词归属或次数错误：{speaker} / {text}")


def validate_corpus(corpus_path=CORPUS):
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if corpus.get("style_profile") in {"suspense-v1", "suspense-voxcpm-v2"}:
        from prepare_style_data import validate_style_corpus

        return validate_style_corpus(corpus_path)
    samples = corpus["samples"]
    prompts = load_prompts()
    isolation_prompts = prompts + benchmark_prompts()
    expected_counts = corpus.get("expected_counts", EXPECTED)
    if set(expected_counts) != {"train", "val"} or any(
        set(counts) != set(EXPECTED[split])
        or any(type(n) is not int or n <= 0 for n in counts.values())
        for split, counts in expected_counts.items()
    ):
        raise ValueError("数据类别数量声明无效")
    system = next(x for x in prompts if x["category"] == "tts_rewrite")["messages"][0]["content"]
    ids, inputs, sources = set(), set(), {"train": set(), "val": set()}
    for sample in samples:
        identifier = sample["id"]
        if not identifier or identifier in ids or sample["split"] not in EXPECTED:
            raise ValueError(f"ID 重复或 split 无效：{identifier}")
        ids.add(identifier)
        sources[sample["split"]].add(sample["source_id"])
        messages = sample["messages"]
        if [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{identifier}: 需要 system/user/assistant 三条消息")
        if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
            raise ValueError(f"{identifier}: 消息为空或不是文本")
        user, answer = messages[1]["content"], messages[2]["content"]
        if user in inputs or any(
            user == case["messages"][-1]["content"] for case in isolation_prompts
        ):
            raise ValueError(f"{identifier}: 重复输入或固定测试输入泄漏")
        inputs.add(user)
        if "<think>" in answer or "```" in answer:
            raise ValueError(f"{identifier}: 答案包含思考块或代码围栏")
        if sample["category"] == "general":
            if any(TAG.match(line) for line in answer.splitlines()):
                raise ValueError(f"{identifier}: 普通任务不能套用演绎标签")
            continue
        if messages[0]["content"] != system or user != "改写下面的文本：\n" + sample["source_text"]:
            raise ValueError(f"{identifier}: 改写指令或来源文本不匹配")
        for line in answer.splitlines():
            match = TAG.fullmatch(line)
            if not match:
                raise ValueError(f"{identifier}: 无效演绎行：{line!r}")
            role, tone, body = match.groups()
            if role not in sample["allowed_roles"] or tone not in {"平静", "低声", "疑问", "大喊"}:
                raise ValueError(f"{identifier}: 未知角色或语气")
            if sample["category"] in {"inner_thought", "no_dialogue"} and role != "旁白":
                raise ValueError(f"{identifier}: 非对白内容被分配给人物")
            # 本轮答案采用原文摘录；只允许拆行时把结尾逗号替换成句号。
            if body.rstrip("。！？?!，,；;：:") not in sample["source_text"]:
                raise ValueError(f"{identifier}: 正文包含原文之外的文字：{body}")
        validate_content_checks(sample)
        for case in isolation_prompts:
            if case["category"] != "tts_rewrite":
                continue
            test_text = case["messages"][-1]["content"].split("\n", 1)[-1]
            matcher = SequenceMatcher(None, sample["source_text"], test_text, autojunk=False)
            if matcher.ratio() > 0.65 or matcher.find_longest_match().size >= 20:
                raise ValueError(f"{identifier}: 与 {case['id']} 过于相似，需复核来源")
    if sources["train"] & sources["val"]:
        raise ValueError("同一 source_id 同时出现在训练集与验证集")
    if (sources["train"] | sources["val"]) & {p.get("source_id") for p in isolation_prompts}:
        raise ValueError("训练或验证故事来源与评估集相交")
    for split, expected in expected_counts.items():
        if Counter(s["category"] for s in samples if s["split"] == split) != expected:
            raise ValueError(f"{split}: 类别数量与约定不符")
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    for left in train:
        for right in val:
            if (
                SequenceMatcher(
                    None, left["source_text"], right["source_text"], autojunk=False
                ).ratio()
                > 0.65
            ):
                raise ValueError(f"训练/验证文本过于相似：{left['id']} / {right['id']}")
    pilot = [{"messages": s["messages"]} for s in samples if s["review"] == "user_approved_pilot"]
    if len(pilot) != 5 or canonical_hash(pilot) != corpus["pilot_records_sha256"]:
        raise ValueError("已批准的 5 条 pilot 内容发生变化")
    return samples, prompts


def export_data(samples, corpus_path=CORPUS):
    """重复执行复用完全相同的数据；拒绝覆盖任何不同内容。"""
    files = {}
    paths = dataset_paths(corpus_path)
    for split in EXPECTED:
        records = [{"messages": s["messages"]} for s in samples if s["split"] == split]
        files[paths[split]] = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
    files[paths["sources"]] = "".join(
        json.dumps({k: v for k, v in s.items() if k != "messages"}, ensure_ascii=False) + "\n"
        for s in samples
    )
    for path, content in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"拒绝覆盖不同内容：{path}；由 agent 处理数据修订和版本管理")
    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def check_exported_data(samples, corpus_path=CORPUS):
    paths = dataset_paths(corpus_path)
    for split in EXPECTED:
        path = paths[split]
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected = [{"messages": s["messages"]} for s in samples if s["split"] == split]
        if actual != expected:
            raise ValueError(f"{path}: 导出数据与语料不一致")


def measure_samples(samples):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "USE_HF"):
        os.environ[key] = "1"
    from swift.model import get_model_processor
    from swift.template import get_template

    model, config = load_model_config(), load_training_config()
    _, processor = get_model_processor(
        str(model.path),
        model_type=model.model_type,
        load_model=False,
        download_model=False,
        use_hf=True,
        device_map="cpu",
        model_kwargs={"local_files_only": True},
    )
    template = get_template(
        processor,
        template_type=model.model_type,
        max_length=config["max_length"],
        truncation_strategy="raise",
        enable_thinking=False,
        loss_scale=config["loss_scale"],
    )
    template.set_mode("train")
    lengths = []
    for sample in samples:
        encoded = template.encode(copy.deepcopy({"messages": sample["messages"]}))
        total, supervised = len(encoded["input_ids"]), sum(x != -100 for x in encoded["labels"])
        if not 0 < supervised <= total <= config["max_length"]:
            raise ValueError(f"{sample['id']}: token 长度或监督范围无效")
        lengths.append(
            {
                "id": sample["id"],
                "split": sample["split"],
                "total_tokens": total,
                "supervised_tokens": supervised,
            }
        )
    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="语料文件；新版本独立导出")
    parser.add_argument("--check-tokens", action="store_true", help="CPU 离线模板检查，不加载权重")
    args = parser.parse_args()
    if json.loads(args.corpus.read_text(encoding="utf-8")).get("style_profile") in {
        "suspense-v1",
        "suspense-voxcpm-v2",
    }:
        from prepare_style_data import prepare

        prepare(args.corpus, args.check_tokens)
        return 0
    samples, prompts = validate_corpus(args.corpus)
    export_data(samples, args.corpus)
    check_exported_data(samples, args.corpus)
    report = {
        "schema_version": 1,
        "corpus_sha256": canonical_hash(samples),
        "prompts_sha256": prompts_fingerprint(prompts),
        "round2_prompts_sha256": prompts_fingerprint(benchmark_prompts())
        if ROUND2_BENCHMARK.exists()
        else None,
        "counts": {
            split: dict(Counter(s["category"] for s in samples if s["split"] == split))
            for split in EXPECTED
        },
        "near_duplicate_check": "exact + SequenceMatcher heuristic; agent source review also required",
    }
    if args.check_tokens:
        report.update(
            template=load_model_config().model_type,
            training=load_training_config(),
            lengths=measure_samples(samples),
        )
        output = dataset_paths(args.corpus)["checks"]
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"[PASS] {sum(s['split'] == 'train' for s in samples)} 条训练 + "
        f"{sum(s['split'] == 'val' for s in samples)} 条验证；pilot 保留；来源、格式和测试隔离通过。"
    )


if __name__ == "__main__":
    main()
