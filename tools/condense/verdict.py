#!/usr/bin/env python3.12
"""Print the head-to-head: Hawking condensed (degradation vs its f16) vs llama Q4_K (+2.1%).
Usage: verdict.py <f16_ppl> <condensed_ppl> <bpw> <label>
"""
import sys
hf, hc, bpw, lbl = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
d = (hc / hf - 1) * 100
print(f"  Hawking {lbl} (~{bpw} bpw): +{d:.1f}%   (ppl {hc:.1f} vs f16 {hf:.1f})")
print(f"  llama.cpp Q4_K_M  (4.5 bpw): +2.1%")
print("  => " + ("🏆 WIN: denser AND >= Q4_K quality" if d < 2.1
                 else f"+{d:.1f}% vs 2.1% — closer, keep tuning"))
