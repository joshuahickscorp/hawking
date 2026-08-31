#!/usr/bin/env python3
"""Fail-closed, read-mostly ModelLake specimen fidelity auditor."""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

STATE = ("MANIFEST_ONLY", "PARTIAL", "CONTENT_COMPLETE", "VERIFYING", "READY_COLD", "CORRUPT_OR_INCOMPLETE")
SEAL = "MODEL_LAKE_SPECIMEN_SEAL.json"
SIDECAR = "MODEL_LAKE_SPECIMEN_SEAL.sha256"

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def safetensor_check(p, index_tensors=()):
    out={"path":str(p),"ok":False,"tensors":[],"representative_reads":0,"failure":None}
    try:
        n=os.path.getsize(p)
        with open(p,"rb") as f:
            raw=f.read(8)
            if len(raw)!=8: raise ValueError("missing 8-byte safetensors header length")
            hl=int.from_bytes(raw,"little")
            if hl>n-8: raise ValueError("header exceeds physical file size")
            obj=json.loads(f.read(hl))
        if not isinstance(obj,dict): raise ValueError("header is not an object")
        data0=8+hl
        for name, spec in obj.items():
            if name=="__metadata__": continue
            if not isinstance(spec,dict) or not isinstance(spec.get("data_offsets"),list) or len(spec["data_offsets"])!=2: raise ValueError("malformed offsets for "+name)
            a,b=spec["data_offsets"]
            if not isinstance(a,int) or not isinstance(b,int) or a<0 or b<a or data0+b>n: raise ValueError("out-of-bounds offsets for "+name)
            out["tensors"].append(name)
        missing=sorted(set(index_tensors)-set(out["tensors"]))
        if missing: raise ValueError("index tensors absent: "+", ".join(missing[:5]))
        # bounded physical representative read, never whole-tensor loading
        for name in out["tensors"][:3]:
            a,b=obj[name]["data_offsets"]
            if b>a:
                with open(p,"rb") as f: f.seek(data0+a); f.read(min(16,b-a))
                out["representative_reads"]+=1
        out["ok"]=True
    except Exception as e: out["failure"]=str(e)
    return out

def parse_asset(p):
    try:
        s=p.suffix.lower()
        if s==".json": json.loads(p.read_text())
        elif s==".py": compile(p.read_text(errors="strict"),str(p),"exec")
        else: return {"parsed":False,"method":"not_applicable"}
        return {"parsed":True,"method":"json" if s==".json" else "python_ast_compile"}
    except Exception as e: return {"parsed":False,"failure":str(e)}

def load_manifest(root, manifest_dir):
    cand=sorted(Path(manifest_dir).glob(root.name+".json"))
    if not cand: return None
    try: return json.loads(cand[0].read_text())
    except Exception: return None

def audit(root, manifest_dir):
    name=root.name; m=load_manifest(root,manifest_dir) or {}
    repo=m.get("repository_id") or m.get("repo_id") or name.split("@")[0].replace("--","/")
    rev=m.get("pinned_revision") or m.get("revision") or (name.split("@",1)[1] if "@" in name else None)
    files=m.get("files") or m.get("file_inventory") or m.get("expected_file_inventory") or []
    inventory_authoritative=bool(files)
    if isinstance(files,dict): files=[dict(v,relative_path=k) if isinstance(v,dict) else {"relative_path":k,"expected_size":v} for k,v in files.items()]
    if not files and root.is_dir():
        # Legacy manifests are intentionally not treated as authoritative inventories.
        # We still hash the bytes present so the seal records evidence, but remains VERIFYING.
        files=[{"relative_path":str(p.relative_to(root))} for p in root.rglob("*") if p.is_file() and p.name not in {SEAL,SIDECAR} and ".cache" not in p.parts]
    expected={x.get("relative_path") or x.get("path"):(x if isinstance(x,dict) else {}) for x in files if isinstance(x,dict)}
    ledger=[]; failures=[]; total=verified=0; index_maps={}; structural=[]
    for rel, meta in sorted(expected.items()):
        p=root/rel; exp=meta.get("expected_size",meta.get("size")); actual=p.stat().st_size if p.is_file() else None
        total += int(exp or 0); remote={k:meta.get(k) for k in ("sha256","lfs_oid","xet_identity","etag") if meta.get(k)}
        item={"relative_path":rel,"expected_size":exp,"actual_size":actual,"remote_content_identity":remote,"locally_computed_sha256":None,"verification_status":"MISSING"}
        if p.is_file() and (exp is None or actual==exp):
            item["locally_computed_sha256"]=sha256(p); verified += actual
            item["verification_status"]="BYTE_MATCH"
            if remote.get("sha256") and remote["sha256"]!=item["locally_computed_sha256"]: item["verification_status"]="HASH_MISMATCH"; failures.append(rel)
            if p.name.endswith(".index.json"):
                try:
                    ix=json.loads(p.read_text()); wm=ix.get("weight_map",{}); index_maps[rel]=wm
                    if not isinstance(wm,dict): raise ValueError("weight_map is not an object")
                except Exception as e: failures.append(rel+": "+str(e)); item["verification_status"]="STRUCTURAL_FAIL"
        elif p.exists(): item["verification_status"]="SIZE_MISMATCH"; failures.append(rel)
        else: failures.append(rel)
        ledger.append(item)
    # Validate index->shard existence and safetensors structure.
    by_shard={}
    for ix, wm in index_maps.items():
        for tensor, shard in wm.items(): by_shard.setdefault(shard,[]).append(tensor)
        for shard in set(wm.values()):
            if shard not in expected or not (root/shard).is_file(): failures.append(ix+" -> missing shard "+shard)
    for shard, tensors in by_shard.items():
        if shard.endswith(".safetensors"):
            c=safetensor_check(root/shard,tensors); structural.append(c)
            if not c["ok"]: failures.append(shard+": "+(c["failure"] or "structural failure"))
    for item in ledger:
        if Path(item["relative_path"]).name in {"config.json","tokenizer.json","tokenizer_config.json","processor_config.json","preprocessor_config.json","generation_config.json"} and item["verification_status"]=="BYTE_MATCH":
            item["structural_parse"]=parse_asset(root/item["relative_path"])
            if not item["structural_parse"].get("parsed") and item["structural_parse"].get("method")!="not_applicable": failures.append(item["relative_path"]+": parse failed")
    content_complete=inventory_authoritative and not failures and all(x["verification_status"]=="BYTE_MATCH" for x in ledger)
    ready=content_complete and bool(repo and rev) and all(x.get("structural_parse",{}).get("parsed",True) for x in ledger) and all(x.get("ok",True) for x in structural)
    status="READY_COLD" if ready else ("CONTENT_COMPLETE" if content_complete else ("VERIFYING" if ledger else "MANIFEST_ONLY"))
    seal={"schema":"MODEL_LAKE_SPECIMEN_SEAL.v1","specimen_identity":{"repository_id":repo,"pinned_revision":rev,"resolved_commit_sha":rev if re.fullmatch(r"[0-9a-f]{40}",str(rev or "")) else None,"acquisition_timestamp":m.get("acquisition_timestamp")},"expected_file_inventory":sorted(expected),"expected_required_files":sorted(expected),"intentionally_excluded_files":[SEAL,SIDECAR],"byte_ledger":ledger,"remote_authority":{"manifest_present":bool(m),"remote_digest_limitation":"Only identities captured in the upstream/cache manifest are comparable; absent remote digests are not source-verified.","cryptographically_source_verified":all(bool(x["remote_content_identity"].get("sha256")) for x in ledger)},"structural_checks":{"safetensors":structural,"indexes":list(index_maps)},"failures":failures,"final_status":status,"seal_digest":{"algorithm":"SHA256","sidecar":SIDECAR,"covers":"canonical JSON bytes with stable key ordering"}}
    raw=json.dumps(seal,sort_keys=True,indent=2).encode()+b"\n"; root.joinpath(SEAL).write_bytes(raw); root.joinpath(SIDECAR).write_text(hashlib.sha256(raw).hexdigest()+"  "+SEAL+"\n")
    return {"specimen":name,"status":status,"expected_bytes":total,"verified_bytes":verified,"files_expected":len(ledger),"files_verified":sum(x["verification_status"]=="BYTE_MATCH" for x in ledger),"failures":failures,"seal":str(root/SEAL)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="/Volumes/corpdrive/hawking-modellake/specimens"); ap.add_argument("--manifest-dir",default="/Volumes/corpdrive/hawking-modellake/manifests"); ap.add_argument("--specimen",action="append"); a=ap.parse_args()
    roots=[Path(a.root)/x for x in a.specimen] if a.specimen else sorted(p for p in Path(a.root).iterdir() if p.is_dir())
    roots=[p for p in roots if p.is_dir()]
    print(json.dumps({"audited_at":time.time(),"results":[audit(p,a.manifest_dir) for p in roots]},indent=2))
if __name__=="__main__": main()
