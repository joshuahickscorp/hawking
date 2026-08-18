"""G028: does the recorded recipe rebuild the artifact?

Same command, same raw source, same parent. Compare the WEIGHT PAYLOAD, which is what the
runtime consumes -- not the pack report, which carries a wall-clock time and a date and can
never be byte-identical.
"""
import hashlib, json, os, sys
RUNS = "workspace/campaign/records/runs/qwen38-27b"
a, b = "compact-q3attn-r1p2-v1", "compact-rebuild-r1p2-v1"

def seg_hashes(name):
    d = os.path.join(RUNS, name, "segments")
    out = {}
    for f in sorted(os.listdir(d)):
        h = hashlib.sha256()
        with open(os.path.join(d, f), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[f] = h.hexdigest()
    return out

ha, hb = seg_hashes(a), seg_hashes(b)
print(f"{a}: {len(ha)} segments")
print(f"{b}: {len(hb)} segments")
only_a = set(ha) - set(hb); only_b = set(hb) - set(ha)
print(f"names only in first  : {len(only_a)}")
print(f"names only in second : {len(only_b)}")
common = set(ha) & set(hb)
diff = [f for f in sorted(common) if ha[f] != hb[f]]
print(f"common segments      : {len(common)}")
print(f"BYTE-IDENTICAL       : {len(common)-len(diff)}")
print(f"DIFFERING            : {len(diff)}")
for f in diff[:5]:
    print(f"   {f}\n     {ha[f][:16]} vs {hb[f][:16]}")
ba = sum(os.path.getsize(os.path.join(RUNS,a,'segments',f)) for f in ha)
bb = sum(os.path.getsize(os.path.join(RUNS,b,'segments',f)) for f in hb)
print(f"\nsegment bytes: {ba:,} vs {bb:,}  delta {bb-ba:+,}")
for n in (a, b):
    r = os.path.join(RUNS, n)
    files = sorted(f for f in os.listdir(r) if os.path.isfile(os.path.join(r, f)))
    print(f"{n} root files: {files}")
cat_a = hashlib.sha256(open(os.path.join(RUNS,a,'catalog.hq38m20'),'rb').read()).hexdigest()
cat_b = hashlib.sha256(open(os.path.join(RUNS,b,'catalog.hq38m20'),'rb').read()).hexdigest()
print(f"catalog identical: {cat_a==cat_b}  ({cat_a[:16]} vs {cat_b[:16]})")
print()
if not diff and not only_a and not only_b:
    print("VERDICT: the weight payload rebuilds BYTE-IDENTICALLY from the recorded recipe.")
else:
    print("VERDICT: the payload does NOT rebuild byte-identically. Investigate before any")
    print("reproducibility claim.")
