"""无模型回归：标签隔离、对白审阅、对照变量与音频边界停顿。"""

import copy
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from run_voxcpm_audition import build_plan, join_segments, local_service, trim_edges, wav_info
from voxcpm_adapter import compile_segments, fingerprint, load_delivery, parse_script, verify_review


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.delivery = load_delivery()
        self.script = "[旁白][平静]纪舟停在门外。\n[纪舟][低声]别开门。\n[纪舟][低声]别开门。"
        self.roles = ["旁白", "纪舟"]
        self.rows = parse_script(self.script, self.roles, self.delivery["tones"])
        self.case = {
            "id": "test",
            "title": "重复对白",
            "source_text": "纪舟停在门外，说：别开门。随后又说：别开门。",
            "script": self.script,
            "allowed_roles": self.roles,
            "review_status": "agent_checked",
            "provenance": "test",
            "fact_checks": [{"source": "停在门外", "target": "停在门外", "role": "旁白"}],
            "speech_checks": [{"text": "别开门。", "speaker": "纪舟", "count": 2}],
        }
        self.case["review_sha256"] = fingerprint(
            {k: self.case[k] for k in ("source_text", "script", "allowed_roles")}
        )

    def test_reject_control_leak_and_unknown_labels(self):
        for text in [
            "<think>内容</think>\n" + self.script,
            "[新人物][平静]内容。",
            "[旁白][恐惧]内容。",
            "[旁白][平静][laughing]内容。",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_script(text, self.roles, self.delivery["tones"])

    def test_stale_review_and_wrong_speaker_are_rejected(self):
        verify_review(self.case, self.rows)
        stale = copy.deepcopy(self.case)
        stale["source_text"] += "门开了。"
        with self.assertRaises(ValueError):
            verify_review(stale, self.rows)
        wrong = copy.deepcopy(self.rows)
        wrong[-1]["speaker"] = "旁白"
        with self.assertRaises(ValueError):
            verify_review(self.case, wrong)

    def test_merge_preserves_repeated_speech_and_api_contract(self):
        segments = compile_segments(self.rows, {"旁白": "n.wav", "纪舟": "a.wav"}, self.delivery)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1]["request"]["text"], "别开门。别开门。")
        for segment in segments:
            request = segment["request"]
            self.assertNotIn("prompt_text", request)
            self.assertNotIn("[", request["text"])
            self.assertEqual(request["clone_mode"], "controllable")
        with self.assertRaises(ValueError):
            compile_segments(self.rows, {"旁白": "n.wav"}, self.delivery)

    def test_comparisons_use_same_narrator_and_control_changes_only_delivery(self):
        # 只包含旁白的真实对照条件，角色演示不能混入它的变量。
        case = copy.deepcopy(self.case)
        bundle = {
            "cases": [case],
            "comparison_ids": ["test"],
            "provisional_voices": {"旁白": "n.wav", "纪舟": "a.wav"},
        }
        plans = build_plan(bundle, self.delivery, "moderate")
        for entry in plans:
            self.assertEqual({s["request"]["audio_path"] for s in entry["segments"]}, {"n.wav"})
        neutral_text = "".join(s["request"]["text"] for s in plans[1]["segments"])
        controlled_text = "".join(s["request"]["text"] for s in plans[2]["segments"])
        self.assertEqual(neutral_text, controlled_text)

    def test_pcm_join_inserts_only_boundary_silence(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x01\x00" * 12000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.wav"
            first.write_bytes(buffer.getvalue())
            output = root / "joined.wav"
            info = join_segments([first, first], [250, 1500], output)
            self.assertEqual(info["frames"], 30000)
            self.assertEqual(wav_info(output.read_bytes())[0]["duration_seconds"], 1.25)
            with self.assertRaises(ValueError):
                join_segments([first, first], [250], output)

    def test_missing_service_does_not_start_or_occupy_port_by_default(self):
        with (
            patch("run_voxcpm_audition.read_json_url", side_effect=URLError("offline")),
            patch("run_voxcpm_audition.subprocess.Popen") as start,
        ):
            with self.assertRaisesRegex(RuntimeError, "手动执行"):
                with local_service(Path("/unused"), "http://127.0.0.1:8322", Path("/unused.log")):
                    self.fail("不应进入服务上下文")
            start.assert_not_called()

    def test_strength_does_not_turn_steady_passage_into_suspense(self):
        row = [{"speaker": "旁白", "tone": "平静", "text": "阳光照在桌面上。"}]
        instructions = {
            compile_segments(row, {"旁白": "n.wav"}, self.delivery, strength)[0]["request"][
                "control_instruction"
            ]
            for strength in ("mild", "moderate", "strong")
        }
        self.assertEqual(len(instructions), 1)

    def test_mild_merges_narrator_without_changing_text_or_speakers(self):
        rows = parse_script(
            "[旁白][冷静]灯还亮着。\n[旁白][紧绷]门外没有回应。\n[纪舟][低声]别开门。",
            self.roles,
            self.delivery["tones"],
        )
        voices = {"旁白": "n.wav", "纪舟": "a.wav"}
        mild = compile_segments(rows, voices, self.delivery, "mild")
        moderate = compile_segments(rows, voices, self.delivery, "moderate")
        self.assertEqual(len(mild), 2)
        self.assertEqual(len(moderate), 3)
        self.assertEqual(
            "".join(s["request"]["text"] for s in mild), "".join(row["text"] for row in rows)
        )
        self.assertEqual(mild[0]["source_tones"], ["冷静", "紧绷"])
        self.assertNotIn("稍慢", mild[0]["request"]["control_instruction"])
        self.assertNotIn("紧张", mild[0]["request"]["control_instruction"])
        self.assertEqual(mild[0]["tone"], "平静")
        self.assertEqual(mild[1]["request"]["audio_path"], "a.wav")

    def test_turn_timing_distinguishes_dialogue_exchange_and_narration(self):
        rows = parse_script(
            "[旁白][平静]纪舟停在门外。\n[纪舟][低声]别开门。\n[袁星][疑问]为什么？\n[旁白][平静]纪舟没有回答。",
            ["旁白", "纪舟", "袁星"],
            self.delivery["tones"],
        )
        segments = compile_segments(
            rows, {"旁白": "n.wav", "纪舟": "a.wav", "袁星": "b.wav"}, self.delivery
        )
        self.assertEqual([s["pause_after_ms"] for s in segments], [400, 650, 500, 0])
        self.assertTrue(all(s["pause_mode"] == "target_gap" for s in segments))
        self.assertEqual(segments[1]["request"]["text"], "别开门。")

    def test_gentle_variation_keeps_continuous_narration_and_steady_scenes(self):
        voices = {"旁白": "n.wav"}
        mixed = parse_script(
            "[旁白][冷静]他看着录像。\n[旁白][迟疑]也许只是镜头移动。\n[旁白][紧绷]刻度线却没变。",
            ["旁白"],
            self.delivery["tones"],
        )
        steady = parse_script(
            "[旁白][平静]茶还温着。\n[旁白][平静]他坐下整理账目。", ["旁白"], self.delivery["tones"]
        )
        segments = compile_segments(mixed, voices, self.delivery)
        self.assertEqual(len(segments), 1)
        self.assertIn("细微起伏", segments[0]["request"]["control_instruction"])
        self.assertNotIn("紧张", segments[0]["request"]["control_instruction"])
        self.assertEqual(
            compile_segments(steady, voices, self.delivery)[0]["request"]["control_instruction"],
            self.delivery["neutral_instruction"],
        )

    def test_strong_localizes_emotion_and_keeps_calm_and_dialogue_restrained(self):
        rows = parse_script(
            "[旁白][冷静]他看着录像。\n[旁白][迟疑]也许镜头动了，\n[旁白][紧绷]刻度线却没变。\n[旁白][冷静]他记下时间。\n[纪舟][迟疑]我只听见椅子动了。",
            self.roles,
            self.delivery["tones"],
        )
        segments = compile_segments(
            rows, {"旁白": "n.wav", "纪舟": "a.wav"}, self.delivery, "strong"
        )
        self.assertEqual([s["tone"] for s in segments], [r["tone"] for r in rows])
        self.assertEqual([s["pause_after_ms"] for s in segments], [350, 180, 350, 400, 0])
        self.assertEqual(
            "".join(s["request"]["text"] for s in segments), "".join(r["text"] for r in rows)
        )
        instructions = [s["request"]["control_instruction"] for s in segments]
        self.assertEqual(instructions[0], instructions[3])
        self.assertEqual(instructions[0], self.delivery["neutral_instruction"])
        self.assertIn("明显犹疑", instructions[1])
        self.assertIn("紧张", instructions[2])
        self.assertIn("日常交谈", instructions[4])
        self.assertNotEqual(instructions[1], instructions[4])
        self.assertTrue(
            all(self.delivery["strength_suffix"]["strong"] not in i for i in instructions)
        )
        self.assertTrue(all(len(s["source_tones"]) == 1 for s in segments))

    def test_total_gap_counts_existing_silence_and_never_cuts_speech(self):
        import array

        rate = 24000
        policy = {"threshold_pcm": 32, "keep_ms": 80, "max_trim_ms": 1200}

        def wav(pcm):
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes(array.array("h", pcm).tobytes())
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root / "a.wav", root / "b.wav", root / "joined.wav"
            first.write_bytes(wav([1000] * 12000 + [0] * 2400))
            second.write_bytes(wav([0] * 2400 + [2000] * 12000))
            info = join_segments([first, second], [650, 0], output, policy, "target_gap")
            boundary = info["boundaries"][0]
            self.assertEqual(boundary["existing_quiet_ms"], 160)
            self.assertEqual(boundary["inserted_ms"], 490)
            self.assertEqual(boundary["total_quiet_ms"], 650)
            pcm = array.array("h", wav_info(output.read_bytes())[1])
            self.assertEqual((pcm.count(1000), pcm.count(2000)), (12000, 12000))
            self.assertEqual(info["duration_seconds"], 1.65)
            # 超过保守裁切上限的尾部空白保留并报告，不为凑目标继续切割。
            first.write_bytes(wav([1000] * 12000 + [0] * (rate * 3)))
            info = join_segments([first, second], [650, 0], output, policy, "target_gap")
            self.assertEqual(info["boundaries"][0]["inserted_ms"], 0)
            self.assertEqual(info["boundaries"][0]["total_quiet_ms"], 1880)

    def test_punctuation_revision_keeps_content_order_and_direct_speech(self):
        import json

        from apply_voxcpm_feedback import REVISED_AUDITION

        for case in json.loads(REVISED_AUDITION.read_text())["cases"]:
            rows = parse_script(case["script"], case["allowed_roles"], self.delivery["tones"])
            verify_review(case, rows)
            if "original_script" not in case:
                continue
            changed = copy.deepcopy(case)
            changed["script"] = changed["script"].replace(
                rows[0]["text"], "突然" + rows[0]["text"], 1
            )
            changed["review_sha256"] = fingerprint(
                {k: changed[k] for k in ("source_text", "script", "allowed_roles")}
            )
            with self.assertRaisesRegex(ValueError, "正文内容或顺序"):
                verify_review(
                    changed,
                    parse_script(
                        changed["script"], changed["allowed_roles"], self.delivery["tones"]
                    ),
                )

    def test_expanded_audition_keeps_real_role_voices_and_reuses_exact_requests_only(self):
        import json

        from apply_voxcpm_feedback import (
            REVISED_AUDITION,
            build_feedback_plan,
            previous_segment_cache,
        )

        bundle = json.loads(REVISED_AUDITION.read_text())
        bundle["level_only_comparison_ids"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_bundle = root / "bundle.json"
            old_bundle.write_text(json.dumps(bundle))
            previous = {
                "bundle_file": str(old_bundle),
                "entries": [
                    {
                        "id": identifier,
                        "variant": "gentle_continuous",
                        "provenance": "test",
                        "audio_file": f"{identifier}/audio.wav",
                        "reused_from": None,
                    }
                    for identifier in bundle["previous_comparison_ids"]
                ],
            }
            plan = build_feedback_plan(bundle, previous, root, self.delivery, "strong")
            self.assertEqual(
                len(plan), len(bundle["cases"]) + len(bundle["previous_comparison_ids"])
            )
            self.assertEqual(sum(not e.get("reuse_audio") for e in plan), len(bundle["cases"]))
            first = next(e for e in plan if not e["optional_comparison"])
            self.assertEqual(len(first["segments"]), 4)
            role_entry = next(e for e in plan if e["id"] == "sus_aud_v5_02")
            for segment in role_entry["segments"]:
                self.assertEqual(
                    segment["request"]["audio_path"],
                    bundle["provisional_voices"][segment["speaker"]],
                )
            folder = root / "sus_test_06"
            folder.mkdir()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(b"\x01\x00" * 12000)
            (folder / "segment-01.wav").write_bytes(buffer.getvalue())
            request = first["segments"][0]["request"]
            (folder / "requests.json").write_text(json.dumps([{"request": request}]))
            self.assertEqual(len(previous_segment_cache(previous, root, plan)), 1)
            wrong = {**request, "audio_path": "wrong-voice.wav"}
            (folder / "requests.json").write_text(json.dumps([{"request": wrong}]))
            self.assertEqual(previous_segment_cache(previous, root, plan), {})

    def test_level_only_comparison_keeps_old_requests_and_requires_raw_cache(self):
        import json

        from apply_voxcpm_feedback import (
            REVISED_AUDITION,
            build_feedback_plan,
            previous_segment_cache,
        )

        bundle = json.loads(REVISED_AUDITION.read_text())
        identifier = "sus_aud_v5_02"
        bundle.update(
            audition_order=[identifier],
            previous_comparison_ids=[identifier],
            level_only_comparison_ids=[identifier],
        )
        case = next(c for c in bundle["cases"] if c["id"] == identifier)
        rows = parse_script(case["script"], case["allowed_roles"], self.delivery["tones"])
        old = compile_segments(rows, bundle["provisional_voices"], self.delivery, "mild")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bundle.json").write_text(json.dumps(bundle))
            folder = root / "prior"
            folder.mkdir()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(b"\x01\x00" * 12000)
            for i in range(1, len(old) + 1):
                (folder / f"segment-{i:02}.wav").write_bytes(buffer.getvalue())
            (folder / "requests.json").write_text(json.dumps(old))
            previous = {
                "bundle_file": str(root / "bundle.json"),
                "entries": [
                    {
                        "id": identifier,
                        "audio_file": "prior/audio.wav",
                        "provenance": "test",
                        "audio": {},
                    }
                ],
            }
            plan = build_feedback_plan(bundle, previous, root, self.delivery, "strong")
            ablation = next(e for e in plan if e["variant"] == "level_only_comparison")
            self.assertTrue(ablation["raw_cache_required"])
            self.assertTrue(ablation["optional_comparison"])
            self.assertEqual(ablation["segments"], old)
            cache = previous_segment_cache(previous, root, plan)
            self.assertTrue(all(fingerprint(s["request"]) in cache for s in ablation["segments"]))

    def test_edge_trim_preserves_quiet_speech_and_internal_pause(self):
        import array

        policy = {"threshold_pcm": 32, "keep_ms": 80, "max_trim_ms": 1200}
        pcm = array.array(
            "h", [0] * 2400 + [40] * 400 + [0] * 800 + [1000] * 400 + [0] * 12000
        ).tobytes()
        adjusted, cuts = trim_edges(pcm, 24000, policy)
        result = array.array("h", adjusted)
        self.assertEqual(cuts, {"head_frames": 480, "tail_frames": 10080})
        self.assertEqual(result.count(40), 400)
        self.assertEqual(result.count(1000), 400)
        self.assertEqual(result[2320:3120], array.array("h", [0] * 800))

    def test_extended_digital_silence_is_removed_without_cutting_weak_or_internal_audio(self):
        import array

        policy = {"threshold_pcm": 32, "keep_ms": 80, "max_trim_ms": 1200, "trim_exact_zeros": True}
        rate = 24000
        samples = [0] * (rate * 3) + [40] * 400 + [0] * 24000 + [1000] * 400 + [0] * (rate * 10)
        adjusted, cuts = trim_edges(array.array("h", samples).tobytes(), rate, policy)
        result = array.array("h", adjusted)
        self.assertEqual(cuts, {"head_frames": rate * 3 - 1920, "tail_frames": rate * 10 - 1920})
        self.assertEqual(
            list(result), [0] * 1920 + [40] * 400 + [0] * 24000 + [1000] * 400 + [0] * 1920
        )
        # 极低幅度的非零尾声保留上限，不能把它视作纯数字静音全部裁掉。
        weak = array.array("h", [1000] * 400 + [25] * (rate * 10)).tobytes()
        adjusted, cuts = trim_edges(weak, rate, policy)
        self.assertEqual(cuts["tail_frames"], rate * 1200 // 1000)
        self.assertEqual(array.array("h", adjusted).count(25), rate * 10 - cuts["tail_frames"])

    def test_default_strength_follows_saved_configuration(self):
        self.assertEqual(self.delivery["default_strength"], "mild")
        default = compile_segments(self.rows, {"旁白": "n.wav", "纪舟": "a.wav"}, self.delivery)
        explicit = compile_segments(
            self.rows, {"旁白": "n.wav", "纪舟": "a.wav"}, self.delivery, "mild"
        )
        self.assertEqual(default, explicit)

    def test_feedback_form_defaults_and_saved_choices_are_preserved(self):
        import json

        from view_voxcpm_results import decision_widget

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("view_voxcpm_results.latest", return_value=root),
                patch("IPython.display.display") as display,
            ):
                decision_widget()
                fields = display.call_args.args[0].children
                self.assertEqual(fields[0].value, "mild")
                self.assertEqual(fields[2].value, "")
                (root / "manifest.json").write_text(json.dumps({"resolved_strength": "strong"}))
                decision_widget()
                fields = display.call_args.args[0].children
                self.assertEqual(fields[0].value, "strong")
                self.assertEqual(fields[2].value, "")
                saved = {"strength": "strong", "verdict": "continue", "comments": "测试保存的意见"}
                (root / "user_decision.json").write_text(json.dumps(saved))
                decision_widget()
                fields = display.call_args.args[0].children
                self.assertEqual(
                    (fields[0].value, fields[1].value, fields[2].value),
                    ("strong", "continue", "测试保存的意见"),
                )


if __name__ == "__main__":
    unittest.main()
