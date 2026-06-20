# tools/headbank/

Stages Eagle5 spec-decode heads from a Drive export into local `$HAWKING_HOME/headbank/<slug>/` directories and emits the runtime env block.

The Colab notebook `colab/maximal_spec_headbank_500u.ipynb` trains heads for each model hawking serves and emits a `headbank_manifest.json` indexing every head, AWQ scales, and runtime profile.

## Usage

```sh
# List available models in the bank
python3 tools/headbank/pull.py \
    --manifest /path/to/hawking_export/headbank_500u_v2/headbank_manifest.json \
    --list

# Stage a head (e.g. Qwen-7B: tau 7.76, 97.7% depth-1 accept)
python3 tools/headbank/pull.py \
    --manifest /path/to/hawking_export/headbank_500u_v2/headbank_manifest.json \
    --slug q7b \
    --env-file ~/.hawking/q7b.env

# Source the env and bench
source ~/.hawking/q7b.env
./target/release/hawking bench --prompt 'why is the sky blue?'
```

## Layout produced

```
$HAWKING_HOME/          (defaults to ~/.hawking)
  headbank/
    q3b/
      head.safetensors    # Eagle5 head
      awq_smoothing.json  # AWQ scales (if present)
      runtime_profile.json
    q7b/
      ...
```

The `runtime_profile.json` `runtime_env` block contains every `HAWKING_*` / `EAGLE5_*` env var the runtime expects.

## Manifest schema

`hawking-headbank-manifest-v1` (produced by the Colab notebook):

```json
{
  "schema": "hawking-headbank-manifest-v1",
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
2. `<manifest_dir>/<slug>/<expected-filename>` (works from a local Drive export).
3. A scan of `<manifest_dir>/<slug>/heads/*.safetensors` as a last resort.
