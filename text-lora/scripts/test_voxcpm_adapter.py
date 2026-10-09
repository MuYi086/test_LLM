"""无模型回归：标签隔离、对白审阅、对照变量与音频边界停顿。"""

import copy
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from run_voxcpm_audition import build_plan, join_segments, local_service, wav_info
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


if __name__ == "__main__":
    unittest.main()
