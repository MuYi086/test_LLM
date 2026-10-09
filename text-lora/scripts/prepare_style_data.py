"""检查、导出独立悬疑风格语料；CPU 模板检查不加载模型权重。"""

import argparse
import copy
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from check_environment import PROJECT_ROOT
from prepare_training_data import (
    TAG,
    benchmark_prompts,
    canonical_hash,
    check_exported_data,
    dataset_paths,
    export_data,
    load_training_config,
    measure_samples,
)
from run_inference import load_prompts, prompts_fingerprint

CORPUS = PROJECT_ROOT / "datasets/suspense_samples_v1.json"
BENCHMARK = PROJECT_ROOT / "eval/suspense_benchmark_v1.json"
PROMPTS = PROJECT_ROOT / "eval/suspense_prompts_v1.jsonl"
TONES = {"平静", "冷静", "压低", "紧绷", "低声", "迟疑", "疑问", "急促", "颤声", "大喊"}
CATEGORIES = {"actions", "three_roles", "inner_thought", "no_dialogue", "ambiguity", "general"}


def validate_answer(case, system, answer):
    """核对明确声明的约束；改写后的完整语义与氛围还须阅读原文。"""
    identifier = case["id"]
    if not isinstance(answer, str) or not answer.strip() or "<think>" in answer or "```" in answer:
        raise ValueError(f"{identifier}: 答案为空或夹带思考块/围栏")
    if case["category"] in {"general", "general_control"}:
        if any(TAG.fullmatch(line) for line in answer.splitlines()):
            raise ValueError(f"{identifier}: 普通任务使用了演绎标签")
        return
    if case["messages"][0]["content"] != system:
        raise ValueError(f"{identifier}: 风格 system 指令不一致")
    if case["messages"][1]["content"] != "改写下面的文本：\n" + case["source_text"]:
        raise ValueError(f"{identifier}: 原文与 user 指令不一致")
    lines = [TAG.fullmatch(line) for line in answer.splitlines()]
    if not lines or any(not line for line in lines):
        raise ValueError(f"{identifier}: 无效演绎行")
    for line in lines:
        if line.group(1) not in case["allowed_roles"] or line.group(2) not in TONES:
            raise ValueError(f"{identifier}: 未知角色或语气")
        if case["category"] in {"inner_thought", "no_dialogue"} and line.group(1) != "旁白":
            raise ValueError(f"{identifier}: 没有直接对白却使用人物标签")
        if case.get("tone_policy") == "steady" and line.group(2) not in {"平静", "冷静"}:
            raise ValueError(f"{identifier}: 平稳反例被强行渲染紧张")
    facts = case.get("fact_checks", [])
    if len(facts) < 3:
        raise ValueError(f"{identifier}: 至少声明三个事实检查片段")
    for fact in facts:
        if not fact["source"] or fact["source"] not in case["source_text"]:
            raise ValueError(f"{identifier}: 事实声明没有来源")
        text = "\n".join(
            line.group(3) for line in lines if fact.get("role") in (None, line.group(1))
        )
        if not fact["target"] or fact["target"] not in text:
            raise ValueError(f"{identifier}: 目标事实缺失或归属错误：{fact}")
    for speech in case.get("speech_checks", []):
        text, speaker, count = speech["text"], speech["speaker"], speech["count"]
        actual = sum(line.group(1) == speaker and line.group(3) == text for line in lines)
        if (
            actual != count
            or answer.count(text) != count
            or case["source_text"].count(text) != count
        ):
            raise ValueError(f"{identifier}: 直接对白归属/次数不一致：{speech}")


def evaluation_paths(corpus_path=CORPUS):
    """每版语料绑定自己的固定评估集，旧版继续使用原有文件。"""
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    evaluation = corpus.get("evaluation", {})
    paths = tuple(
        PROJECT_ROOT / evaluation.get(key, str(default.relative_to(PROJECT_ROOT)))
        for key, default in (("benchmark", BENCHMARK), ("prompts", PROMPTS))
    )
    if any(not path.resolve().is_relative_to(PROJECT_ROOT) for path in paths):
        raise ValueError("评估路径不能超出项目")
    return paths


def read_benchmark(path=BENCHMARK):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"]
    if len(cases) != 10 or Counter(c["category"] for c in cases) != {
        "tts_rewrite": 8,
        "general_control": 2,
    }:
        raise ValueError("风格测试应为 8 条改写与 2 条普通任务")
    if len({c["id"] for c in cases}) != 10 or len({c["source_id"] for c in cases}) != 10:
        raise ValueError("风格测试 ID 或故事来源重复")
    for case in cases:
        if [m["role"] for m in case["messages"]] != ["system", "user"]:
            raise ValueError("测试输入只能包含 system/user")
        validate_answer(case, data["rewrite_system"], case["reference_output"])
    return data


def prompt_records(benchmark_path=BENCHMARK):
    return [
        {
            key: copy.deepcopy(case[key])
            for key in ("id", "source_id", "category", "messages", "review_criteria")
        }
        for case in read_benchmark(benchmark_path)["cases"]
    ]


def validate_style_corpus(corpus_path=CORPUS):
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if (corpus.get("style_profile"), corpus["dataset_version"]) not in {
        ("suspense-v1", "teaching-v3"),
        ("suspense-voxcpm-v2", "teaching-v4"),
    }:
        raise ValueError("需要已登记的独立悬疑风格语料版本")
    benchmark_path, _ = evaluation_paths(corpus_path)
    benchmark = read_benchmark(benchmark_path)
    if corpus["rewrite_system"] != benchmark["rewrite_system"]:
        raise ValueError("训练与风格测试指令不一致")
    samples, expected = corpus["samples"], corpus["expected_counts"]
    if set(expected) != {"train", "val"} or any(
        set(counts) != CATEGORIES or any(type(n) is not int or n <= 0 for n in counts.values())
        for counts in expected.values()
    ):
        raise ValueError("风格数据类别声明无效")
    old_tests = load_prompts() + benchmark_prompts()
    tests = benchmark["cases"] + old_tests + read_benchmark()["cases"]
    ids, inputs, source_splits = set(), set(), {}
    for case in samples:
        identifier = case["id"]
        if identifier in ids or case["split"] not in expected:
            raise ValueError(f"ID 重复或 split 错误：{identifier}")
        ids.add(identifier)
        previous = source_splits.setdefault(case["source_id"], case["split"])
        if previous != case["split"] or case["source_id"] in {t.get("source_id") for t in tests}:
            raise ValueError(f"{identifier}: 来源跨集合")
        messages = case["messages"]
        if [m["role"] for m in messages] != ["system", "user", "assistant"] or any(
            not isinstance(m["content"], str) or not m["content"].strip() for m in messages
        ):
            raise ValueError(f"{identifier}: 消息结构错误")
        if messages[1]["content"] in inputs or any(
            messages[1]["content"] == t["messages"][-1]["content"] for t in tests
        ):
            raise ValueError(f"{identifier}: 输入重复或测试泄漏")
        inputs.add(messages[1]["content"])
        validate_answer(case, corpus["rewrite_system"], messages[-1]["content"])
        for test in tests:
            if test["category"] != "tts_rewrite" or case["category"] == "general":
                continue
            text = test.get("source_text", test["messages"][-1]["content"].split("\n", 1)[-1])
            match = SequenceMatcher(None, case["source_text"], text, autojunk=False)
            if match.ratio() > 0.65 or match.find_longest_match().size >= 20:
                raise ValueError(f"{identifier}: 与测试 {test['id']} 过于相似")
    for split, counts in expected.items():
        if Counter(c["category"] for c in samples if c["split"] == split) != counts:
            raise ValueError(f"{split}: 类别数量不匹配")
    train = [c for c in samples if c["split"] == "train"]
    val = [c for c in samples if c["split"] == "val"]
    for left in train:
        for right in val:
            if (
                SequenceMatcher(
                    None, left["source_text"], right["source_text"], autojunk=False
                ).ratio()
                > 0.65
            ):
                raise ValueError("风格训练/验证文本过于相似")
    return samples, prompt_records(benchmark_path)


def export_prompts(benchmark_path=BENCHMARK, prompts_path=PROMPTS):
    content = "".join(
        json.dumps(c, ensure_ascii=False) + "\n" for c in prompt_records(benchmark_path)
    )
    if prompts_path.exists() and prompts_path.read_text(encoding="utf-8") != content:
        raise ValueError("拒绝覆盖已经固定的风格测试输入")
    if not prompts_path.exists():
        prompts_path.write_text(content, encoding="utf-8")
    if load_prompts(prompts_path) != prompt_records(benchmark_path):
        raise ValueError("风格测试输入导出不一致")


def prepare(corpus_path=CORPUS, check_tokens=False):
    samples, prompts = validate_style_corpus(corpus_path)
    export_data(samples, corpus_path)
    check_exported_data(samples, corpus_path)
    export_prompts(*evaluation_paths(corpus_path))
    report = {
        "schema_version": 1,
        "style_profile": json.loads(Path(corpus_path).read_text(encoding="utf-8"))["style_profile"],
        "corpus_sha256": canonical_hash(samples),
        "prompts_sha256": prompts_fingerprint(prompts),
        "counts": {
            split: dict(Counter(s["category"] for s in samples if s["split"] == split))
            for split in ("train", "val")
        },
        "review": "agent authored and checked; user style decision pending",
        "checks_scope": "格式、声明的事实/对白与来源；不能自动证明语义或声音氛围",
    }
    if check_tokens:
        report.update(training=load_training_config(), lengths=measure_samples(samples))
        dataset_paths(corpus_path)["checks"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("[PASS] 悬疑风格数据、目标检查与独立测试导出完成；原有数据保留。", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--check-tokens", action="store_true")
    args = parser.parse_args()
    prepare(args.corpus, args.check_tokens)


if __name__ == "__main__":
    main()
