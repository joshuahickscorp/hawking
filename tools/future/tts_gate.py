"""The failure a TTS output gate does not see: a decoder stuck on one code.

The Higgs sensitivity rungs were scored by three gates -- energy (silence and
dropout), ASR (truncation, looping, word error) and duration. Measuring the
audio directly showed 21 of 24 units run to the token cap with an envelope that
locks to a CONSTANT rms for 14-30 s after ~10 s of real speech. Two of the three
gates are blind to it by construction:

  silent/dropout  the buzz sits at rms 0.22-0.55, four orders above the 1e-4
                  silence threshold, so no window is ever counted silent
  looped/wer      it is not speech, so ASR transcribes no extra words; spoken
                  fraction stays at 1.0 and word error stays low

Only the duration gate fires, and only because a stuck unit runs to the cap. So
the cheapest-looking gate was the sole detector of the dominant failure, and the
two gates the module docstring calls the real ones saw nothing. That is worth a
direct detector: duration catches this by side effect, and stops catching it the
moment a stuck unit happens to emit EOS near the expected length.

Detect it where it lives -- in the envelope. Real speech varies frame to frame;
a fixed point does not.
"""
from __future__ import annotations
import numpy as np


def stuck_run(audio, sr: int, frame_ms: int = 250, rtol: float = 0.06,
              floor: float = 1e-3) -> dict:
    """Longest run of consecutive frames whose rms is flat within `rtol`.

    `floor` excludes silence: a silent tail is also flat, but it is a different
    failure and `energy()` already names it. Returns run_s and its onset, so a
    caller can gate on the run and trim at the onset.
    """
    a = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = max(1, int(sr * frame_ms / 1000))
    trimmed = a[: len(a) - len(a) % n]
    if len(trimmed) < 2 * n:
        return {"stuck_s": 0.0, "onset_s": None, "frames": 0}
    r = np.sqrt((trimmed.reshape(-1, n) ** 2).mean(axis=1))
    best = cur = best_start = cur_start = 0
    for i in range(1, len(r)):
        flat = r[i] > floor and abs(r[i] - r[i - 1]) <= rtol * max(r[i - 1], 1e-9)
        if flat:
            if not cur:
                cur, cur_start = 2, i - 1
            else:
                cur += 1
        else:
            cur = 0
        if cur > best:
            best, best_start = cur, cur_start
    return {"stuck_s": best * frame_ms / 1000,
            "onset_s": best_start * frame_ms / 1000 if best else None,
            "frames": len(r)}


def _demo():
    sr = 24000
    rng = np.random.default_rng(7)

    def speechish(seconds):
        # amplitude modulated at a syllable rate: envelope varies frame to frame
        t = np.arange(int(sr * seconds)) / sr
        env = 0.3 * (1 + np.sin(2 * np.pi * 3.1 * t)) * (0.5 + rng.random(len(t)) * 0.5)
        return (env * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def buzz(seconds, amp=0.45):
        t = np.arange(int(sr * seconds)) / sr
        return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # positive: 10 s speech then 20 s fixed point, the measured Higgs shape
    got = stuck_run(np.concatenate([speechish(10), buzz(20)]), sr)
    assert got["stuck_s"] >= 15.0, got
    assert 9.0 <= got["onset_s"] <= 11.0, got

    # negative control 1: speech alone must not trip
    got = stuck_run(speechish(30), sr)
    assert got["stuck_s"] < 3.0, got

    # negative control 2: silence is flat too, but `floor` leaves it to energy()
    got = stuck_run(np.zeros(sr * 30, dtype=np.float32), sr)
    assert got["stuck_s"] == 0.0, got

    # negative control 3: a slow fade is flat-ish per step but never a fixed
    # point over a long run -- rtol must not swallow gradual change
    t = np.arange(sr * 30) / sr
    fade = ((1 - t / 30) * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    got = stuck_run(fade, sr, rtol=0.001)
    assert got["stuck_s"] < 3.0, got

    print("tts_gate: stuck_run detects the fixed point, clears speech, silence and fades")


if __name__ == "__main__":
    _demo()
