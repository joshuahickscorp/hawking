# Revision 1: verify loaded top-k members against sealed sidecar hashes

The route census implementation and measured distribution are otherwise
acceptable, but the receipt currently binds `array_sha256` values from capsule
sidecars without proving that the loaded top-k array matches them.

For example, the sealed sidecar hash for
`L04_L15.npz:layer_05/topk_indices` is the SHA-256 of the C-contiguous raw
`int32` array bytes.  The census already has that array in memory, so this check
does not require hashing a whole capsule or loading another member.

## Required correction

For every loaded top-k member, including duplicate copies:

1. resolve the exact sidecar key, accepting the established
   `layer_NN/topk_indices` form and an optional `.npy` suffix;
2. compute SHA-256 over the loaded array’s C-contiguous raw bytes in its original
   integer dtype (reshape does not change bytes);
3. require a present 64-hex sealed array hash;
4. fail closed if the computed value differs;
5. record:
   - `sealed_array_sha256`;
   - `computed_array_bytes_sha256`;
   - `sealed_array_hash_verified=true`.

The receipt must publish aggregate proof:

- `n_loaded_topk_members=87`;
- `n_sealed_array_hashes_present=87`;
- `n_sealed_array_hashes_verified=87`;
- `all_loaded_topk_match_sealed_array_hashes=true`.

Canonical-member rows must carry the verified sealed/computed hashes and flag.
Duplicate agreement still uses normalized-array equality and remains separately
reported.

Do not hash whole capsule files.  Continue to bind their already sealed capsule
hashes, and explicitly state `whole_capsule_hash_recomputed=false`.

Tighten the member loader so direct calls accept only the exact
`layer_NN/topk_indices.npy` pattern; an arbitrary path merely ending in
`topk_indices.npy` must fail.

## Tests

Add fake-sidecar tests proving:

- a matching sealed raw-array hash passes;
- a missing hash fails;
- a malformed hash fails;
- a mismatched hash fails;
- both sidecar key suffix variants are accepted;
- all 87 real loaded members are verified in the generated receipt;
- a non-exact member path is refused;
- deterministic regeneration remains byte-identical;
- all fences remain false.

Regenerate JSON/Markdown, rerun focused census tests, v2 tests, both selftests,
and `py_compile`.  Preserve the measured band counts and byte scenarios exactly.
No source body, hidden-state member, or whole capsule hash is authorized.
