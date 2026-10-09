"""CPU音频回归：固定增益保留时间位置，响度趋同且真峰值受限。"""

import array
import math
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from voxcpm_adapter import load_delivery
from voxcpm_audio_levels import (
    adjust_tempo,
    gain_plan,
    join_level_comparison,
    match_segment_level,
    scale_pcm,
)


class AudioLevelTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "需要已有的本机FFmpeg")
    def test_slight_slowdown_extends_speech_and_preserves_pitch_and_source(self):
        rate = 24000
        samples = [round(3000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(rate * 4)]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.wav", Path(directory) / "slower.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes(array.array("h", samples).tobytes())
            original = source.read_bytes()
            processing = adjust_tempo(source, output, 0.94)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(processing["after"]["sample_rate"], rate)
            self.assertAlmostEqual(processing["after"]["duration_seconds"], 4 / 0.94, delta=0.08)
            with wave.open(str(output), "rb") as audio:
                result = array.array("h", audio.readframes(audio.getnframes()))
            middle = result[rate : 2 * rate]
            crossings = sum(a <= 0 < b for a, b in zip(middle[:-1], middle[1:], strict=True))
            self.assertAlmostEqual(crossings, 220, delta=2)
            with self.assertRaisesRegex(ValueError, "不能覆盖"):
                adjust_tempo(source, source, 0.94)

    def test_static_gain_preserves_silence_sample_count_and_relative_dynamics(self):
        pcm = array.array("h", [0, 120, -120, 600, -600, 0]).tobytes()
        result = array.array("h", scale_pcm(pcm, 6.020599913279624))
        self.assertEqual(list(result), [0, 240, -240, 1200, -1200, 0])
        with self.assertRaisesRegex(ValueError, "削波"):
            scale_pcm(array.array("h", [30000]).tobytes(), 6)

    def test_true_peak_takes_priority_over_unreachable_loudness(self):
        policy = load_delivery()["profiles"]["mild"]["loudness"]
        result = gain_plan({"integrated_lufs": -30, "true_peak_dbtp": -3}, -23, policy)
        self.assertEqual(result["requested_gain_db"], 7)
        self.assertEqual(result["applied_gain_db"], 1)
        self.assertTrue(result["limited_by_true_peak"])

    def test_quiet_clip_cannot_receive_the_old_excessive_boost(self):
        policy = load_delivery()["profiles"]["strong"]["loudness"]
        result = gain_plan({"integrated_lufs": -40.08, "true_peak_dbtp": -20}, -26, policy)
        self.assertEqual(result["applied_gain_db"], 12)
        self.assertTrue(result["limited_by_gain_range"])

    def test_level_comparison_preserves_baseline_cut_and_padding_even_for_quiet_edges(self):
        from run_voxcpm_audition import wav_info

        rate = 24000
        samples = [25] * 2400 + [1000] * 12000 + [25] * 2400
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "quiet-edge.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes(array.array("h", samples).tobytes())
            baseline = {
                "sample_rate": rate,
                "frames": (len(samples) - 200) * 2 + 12000,
                "edge_trims": [{"head_frames": 100, "tail_frames": 100}] * 2,
                "tempo_processing": [{"after": {"frames": len(samples)}}] * 2,
                "boundaries": [{"configured_ms": 650, "inserted_ms": 500}],
            }
            output = root / "joined.wav"
            result = join_level_comparison([source, source], output, baseline)
            pcm = array.array("h", wav_info(output.read_bytes())[1])
            expected = samples[100:-100] + [0] * 12000 + samples[100:-100]
            self.assertEqual(list(pcm), expected)
            self.assertEqual(result["frames"], baseline["frames"])
            baseline["tempo_processing"][0]["after"]["frames"] += 1
            with self.assertRaisesRegex(ValueError, "不能改变"):
                join_level_comparison([source, source], output, baseline)

    @unittest.skipUnless(shutil.which("ffmpeg"), "需要已有的本机FFmpeg")
    def test_tone_offsets_preserve_quiet_expression_without_changing_samples_or_timing(self):
        policy = load_delivery()["profiles"]["strong"]["loudness"]
        rate = 24000
        samples = [round(3000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(rate * 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes(array.array("h", samples).tobytes())
            calm = match_segment_level(source, root / "calm.wav", "洛洵", policy, "冷静")
            quiet = match_segment_level(source, root / "quiet.wav", "洛洵", policy, "迟疑")
            self.assertEqual(calm["target_lufs"], -25)
            self.assertEqual(quiet["target_lufs"], -26)
            self.assertAlmostEqual(
                calm["after"]["integrated_lufs"] - quiet["after"]["integrated_lufs"], 1, delta=0.1
            )
            for name in ("calm.wav", "quiet.wav"):
                with wave.open(str(root / name), "rb") as audio:
                    self.assertEqual(
                        (audio.getnframes(), audio.getframerate()), (len(samples), rate)
                    )

    @unittest.skipUnless(shutil.which("ffmpeg"), "需要已有的本机FFmpeg")
    def test_actual_loudness_matching_reduces_role_difference_without_resampling(self):
        policy = load_delivery()["profiles"]["mild"]["loudness"]
        rate = 24000
        levels = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, amplitude in enumerate([600, 6000]):
                source, output = root / f"raw-{index}.wav", root / f"level-{index}.wav"
                samples = (
                    [0] * 2400
                    + [
                        round(amplitude * math.sin(2 * math.pi * 220 * i / rate))
                        for i in range(rate * 2)
                    ]
                    + [0] * 2400
                )
                with wave.open(str(source), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(rate)
                    audio.writeframes(array.array("h", samples).tobytes())
                original = source.read_bytes()
                level = match_segment_level(source, output, "旁白", policy)
                self.assertEqual(source.read_bytes(), original)
                self.assertLessEqual(level["after"]["true_peak_dbtp"], -1.95)
                with wave.open(str(output), "rb") as audio:
                    self.assertEqual(
                        (audio.getframerate(), audio.getnframes()), (rate, len(samples))
                    )
                    adjusted = array.array("h", audio.readframes(audio.getnframes()))
                    self.assertEqual(adjusted[:2400], array.array("h", [0] * 2400))
                    self.assertEqual(adjusted[-2400:], array.array("h", [0] * 2400))
                levels.append(level)
            self.assertGreater(
                abs(
                    levels[0]["before"]["integrated_lufs"] - levels[1]["before"]["integrated_lufs"]
                ),
                19,
            )
            self.assertLess(
                abs(levels[0]["after"]["integrated_lufs"] - levels[1]["after"]["integrated_lufs"]),
                0.2,
            )
            self.assertLess(abs(levels[0]["after"]["integrated_lufs"] + 23), 0.2)


if __name__ == "__main__":
    unittest.main()
