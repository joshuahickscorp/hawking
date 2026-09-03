import numpy as np
N=26_895_998_464
# MEASURED anchors this session
A={
 "G0_uniform_q4":       dict(bytes=14_297_694_680, disp=964,  ns=41_072_084,  cls="fast"),
 "q3mlp":               dict(bytes=13_963_923_935, disp=1156, ns=148_588_917, cls="slow"),
 "q4mse_hgravu01":      dict(bytes=13_897_843_472, disp=1156, ns=186_616_958, cls="slow"),
}
# component split of the fast anchor (sealed ledger)
FAST_ADDR_NS=21_293_103; FAST_ADDR_BYTES=13_611_663_360
REMAINDER=41_072_084-FAST_ADDR_NS      # deltanet+gqa+ceremony+rest, kernel-class independent

bw_fast=FAST_ADDR_BYTES/FAST_ADDR_NS   # B/ns
print(f"fast-class effective bandwidth  {bw_fast*1e9/1e9:.2f} GB/s  (from G0 weight_addressing)")

# FIT slow-class efficiency on ONE slow anchor only
f=A["q4mse_hgravu01"]
slow_addr_ns=f["ns"]-REMAINDER
bw_slow=f["bytes"]/slow_addr_ns
print(f"slow-class effective bandwidth  {bw_slow*1e9/1e9:.2f} GB/s  (fitted on q4mse_hgravu01 ONLY)")
print(f"slow/fast ratio {bw_fast/bw_slow:.2f}x\n")

def predict(b,cls): return b/(bw_fast if cls=="fast" else bw_slow)+REMAINDER
print(f"{'anchor':<20}{'measured ns':>14}{'predicted ns':>14}{'rel err':>10}{'fitted?':>9}")
for k,v in A.items():
    p=predict(v["bytes"],v["cls"]); e=abs(p-v["ns"])/v["ns"]
    fit = "FIT" if k=="q4mse_hgravu01" else "held-out"
    print(f"{k:<20}{v['ns']:>14,}{p:>14,.0f}{e:>10.4f}{fit:>9}")

print("\nwhat the model says about mechanisms, at complete BPW targets:")
for bpw,name in ((4.2527,"G0 today"),(2.0,"unified alloc"),(1.0,"sub-1 target")):
    b=bpw*N/8
    for cls in ("fast","slow"):
        ns=predict(b,cls)
        print(f"  {name:<14} {bpw:>6.4f} BPW  {cls:>4}-class -> {ns/1e6:>7.2f} ms  {1e9/ns:>6.2f} TPS")
print(f"\nfloor even at 0 bytes: remainder {REMAINDER/1e6:.2f} ms -> {1e9/REMAINDER:.1f} TPS ceiling")
