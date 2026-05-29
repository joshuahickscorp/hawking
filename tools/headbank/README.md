# `tools/headbank` — Eagle5 head bank for dismantle

The Colab `colab/maximal_spec_headbank_500u.ipynb` trains a polished Eagle5
spec-decode head for every model dismantle's Rust runtime serves. It emits a
`headbank_manifest.json` indexing every head + AWQ scales + runtime profile.

`pull.py` is the local-side fetcher. Given a manifest path (either a local
copy or a Drive-export folder), it stages the artifacts for one model slug
into `$DISMANTLE_HOME/headbank/<slug>/` and emits the runtime env block.

## Quick start

```bash
# 1. List available models in the bank (point at the manifest INSIDE the
#    downloaded Drive export folder).
python3 tools/headbank/pull.py \
    --manifest /path/to/dismantle_export/headbank_500u_v2/headbank_manifest.json \
    --list

# 2. Stage Qwen-7B (the strongest head: tau 7.76, 97.7% depth-1)
python3 tools/headbank/pull.py \
    --manifest /path/to/dismantle_export/headbank_500u_v2/headbank_manifest.json \
    --slug q7b \
    --env-file ~/.dismantle/q7b.env

# 3. Source the env and bench
source ~/.dismantle/q7b.env
./target/release/dismantle bench --prompt 'why is the sky blue?'
```

> Verified against the real `headbank_500u_v2` Drive export layout: the
> manifest stores Colab-absolute paths, so `pull.py` resolves the head via
> the `<export>/<slug>/heads/*.safetensors` fallback and the AWQ scales +
> runtime profile via `<export>/<slug>/{awq,runtime_profiles}/…`. The staged
> `runtime_profile.json` + env file have `EAGLE5_HEAD` / `DISMANTLE_AWQ_SCALES`
> rewritten to the local copies. One-command-per-model works as shown.

## Layout produced

```
$DISMANTLE_HOME/                       (defaults to ~/.dismantle)
  headbank/
    q3b/
      head.safetensors                 # the Eagle5 head
      awq_smoothing.json               # AWQ scales (if present)
      runtime_profile.json             # patched: paths point at staged copies
    q7b/
      ...
    q05b/
      ...
    dsv2/
      ...
```

The `runtime_profile.json` `runtime_env` block contains every `DISMANTLE_*` /
`EAGLE5_*` env var the runtime expects. Source it and the head is wired.

## Manifest schema

`dismantle-headbank-manifest-v1` (produced by the Colab notebook):

```json
{
  "schema": "dismantle-headbank-manifest-v1",
  "repo_sha": "abc1234",
  "entries": [
    {
      "slug": "q3b",
      "hf_id": "Qwen/Qwen2.5-3B-Instruct",
      "arch": "qwen2",
      "gguf_name": "qwen2.5-3b-instruct-q4_k_m.gguf",
      "profile_name": "qwen3b-instruct-q4k.m3pro18.json",
      "head_path": "<absolute path on Drive>",
      "head_sha256": "...",
      "awq_scales": "<absolute path on Drive>",
      "runtime_profile": "<absolute path on Drive>",
      "metrics": {
        "tau": 7.99,
        "depth1_accept_rate": 0.96,
        "accepted_draft_tokens_per_verify": 23.6,
        "offline_projected_tps": 1869.88,
        "policy_kind": "fixed_k"
      }
    }
  ]
}
```

`pull.py` resolves head/AWQ/profile paths by trying, in order:

1. The absolute path in the manifest (works when run on Colab itself).
2. `<manifest_dir>/<slug>/<expected-filename>` (works when run on the Drive
   export pulled to your laptop).
3. A scan of `<manifest_dir>/<slug>/heads/*.safetensors` as a last resort.

This means you can either run `pull.py` against the manifest file inside the
Drive export directory tree, or copy that whole tree somewhere first and
point at the local copy — both work.
