"""Header-only census: which specimens have an expert organ the screen can find?

Reads each shard's safetensors header (8-byte len + JSON), never a tensor.
"""
import glob, json, re, struct, sys, time, collections

PER_EXPERT = re.compile(r"\.(?:layers|blocks)\.(\d+)\..*?experts?\.(\d+)\.([A-Za-z_0-9]+)\.weight$")
OLD_RE     = re.compile(r"\.layers\.(\d+)\..*experts?\.(\d+)\.(\w+proj)\.weight$")  # shipped tool

def header(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))

def census(snap):
    fs = sorted(glob.glob(snap.rstrip("/") + "/*.safetensors"))
    if not fs:
        return {"shards": 0, "verdict": "NO_SAFETENSORS"}
    cfg = {}
    try: cfg = json.load(open(snap.rstrip("/") + "/config.json"))
    except Exception: pass
    def deep(d, keys):
        for k in keys:
            if k in d: return d[k]
        for v in d.values():
            if isinstance(v, dict):
                r = deep(v, keys)
                if r is not None: return r
        return None
    n_exp = deep(cfg, ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"])
    per, stacked, old = collections.defaultdict(set), [], set()
    for f in fs:
        try: h = header(f)
        except Exception: continue
        for k, meta in h.items():
            if k == "__metadata__": continue
            m = PER_EXPERT.search(k)
            if m: per[int(m.group(1))].add(int(m.group(2)))
            if OLD_RE.search(k): old.add(int(OLD_RE.search(k).group(1)))
            sh = meta.get("shape") or []
            if len(sh) == 3 and sh[0] >= 8 and (n_exp is None or sh[0] == n_exp):
                stacked.append((k, sh))
    first_per = min(per) if per else None
    n_per = len(per[first_per]) if per else 0
    first_stacked = None
    if stacked:
        ls = [int(mm.group(1)) for k, _ in stacked
              if (mm := re.search(r"\.(?:layers|blocks)\.(\d+)\.", k))]
        first_stacked = min(ls) if ls else None
    return {"shards": len(fs), "config_num_experts": n_exp,
            "storage": "per-expert" if per else ("stacked" if stacked else "none"),
            "first_expert_layer": first_per if per else first_stacked,
            "n_experts_seen": n_per or (stacked[0][1][0] if stacked else 0),
            "shipped_tool_layer0_hit": (0 in old),
            "verdict": ("SCREENABLE" if (per or stacked) else "NO_EXPERT_ORGAN")}

if __name__ == "__main__":
    cat = json.load(open("/Users/scammermike/Downloads/hawking/receipts/future/modellake-index/catalog.json"))
    out = []
    for s in sorted(cat["specimens"], key=lambda x: x["bytes"]):
        t0 = time.time(); c = census(s["path"]); c["header_scan_s"] = round(time.time()-t0, 2)
        c.update(slug=s["slug"], family=s["architecture_family"], gb=round(s["bytes"]/2**30, 1))
        out.append(c); print(json.dumps(c), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
