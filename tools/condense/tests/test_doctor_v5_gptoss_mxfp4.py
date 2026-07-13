from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CONDENSE = HERE.parent
ROOT = CONDENSE.parents[1]
sys.path.insert(0, str(CONDENSE))

import doctor_v5_gptoss_moe_adapter as adapter
import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_parameter_manifest as generic_parameter_manifest


class GptOssMxfp4Tests(unittest.TestCase):
    def test_synthetic_decoder_matches_transformers_reference(self) -> None:
        import numpy as np
        import torch
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

        rng = np.random.default_rng(20260713)
        blocks = rng.integers(0, 256, size=(2, 5, 3, 4), dtype=np.uint8)
        scales = rng.integers(120, 136, size=(2, 5, 3), dtype=np.uint8)
        ours_bits = mxfp4.decode_mxfp4_groups_bf16(blocks, scales)
        ours = torch.from_numpy(ours_bits.view(np.int16)).view(torch.bfloat16)
        reference = _convert_moe_packed_tensors(
            torch.from_numpy(blocks), torch.from_numpy(scales),
            dtype=torch.bfloat16, rows_per_chunk=7,
        ).transpose(1, 2).contiguous()
        self.assertEqual(tuple(ours.shape), tuple(reference.shape))
        self.assertTrue(torch.equal(ours.view(torch.int16), reference.view(torch.int16)))

    def test_exhaustive_u8_and_ue8_decoder_matches_reserved_reference_cases(self) -> None:
        import numpy as np
        import torch
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

        blocks = np.broadcast_to(
            np.arange(256, dtype=np.uint8)[:, None, None, None], (256, 256, 1, 1)
        ).copy()
        scales = np.broadcast_to(
            np.arange(256, dtype=np.uint8)[None, :, None], (256, 256, 1)
        ).copy()
        ours = torch.from_numpy(
            mxfp4.decode_mxfp4_groups_bf16(blocks, scales).view(np.int16)
        ).view(torch.bfloat16)
        reference = _convert_moe_packed_tensors(
            torch.from_numpy(blocks), torch.from_numpy(scales),
            dtype=torch.bfloat16, rows_per_chunk=4096,
        ).transpose(1, 2).contiguous()
        self.assertTrue(torch.equal(ours.view(torch.int16), reference.view(torch.int16)))

    @unittest.skipUnless(mxfp4.DEFAULT_MODEL_DIR.is_dir(), "local GPT-OSS source absent")
    def test_live_header_inventory_uses_logical_denominator(self) -> None:
        inspection = mxfp4.inspect_model()
        inventory = mxfp4.build_inventory(inspection)
        self.assertEqual([], mxfp4.validate_inventory(inventory))
        accounting = inventory["parameter_accounting"]
        self.assertEqual(116_829_156_672, accounting["logical_model_parameters"])
        self.assertEqual(
            63_081_444_672,
            accounting["serialized_safetensors_elements_not_a_parameter_denominator"],
        )
        self.assertEqual(3_583_180_800,
                         accounting["ue8_scale_elements_side_information_not_parameters"])
        self.assertEqual(5_132_852_352,
                         accounting["active_compute_parameter_equivalent"])
        self.assertAlmostEqual(
            4.467981630796993,
            inventory["physical_accounting"][
                "native_tensor_payload_bpw_over_logical_parameters"
            ],
        )

    @unittest.skipUnless(mxfp4.DEFAULT_MODEL_DIR.is_dir(), "local GPT-OSS source absent")
    def test_one_expert_stage_is_exact_and_source_safe(self) -> None:
        import torch
        from safetensors import safe_open
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

        inspection = mxfp4.inspect_model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expert.safetensors"
            receipt = mxfp4.stage_expert_batch(
                inspection, layer=0, expert_start=0, expert_count=1, output=output,
            )
            self.assertFalse(receipt["lifecycle"]["source_files_deleted"])
            self.assertEqual([], mxfp4.validate_staging_receipt(
                receipt, verify_files=True
            ))
            for binding in receipt["source_bindings"]:
                for extent in binding["source_extents"]:
                    self.assertRegex(extent["range_sha256"], r"^[0-9a-f]{64}$")
            self.assertLessEqual(
                receipt["memory_contract"]["maximum_raw_source_slice_bytes"],
                8_294_400,
            )
            index = json.loads((inspection.weights_root /
                                "model.safetensors.index.json").read_text())
            with safe_open(str(output), framework="pt", device="cpu") as staged:
                for projection in ("mlp1", "mlp2"):
                    block_name = f"block.0.mlp.{projection}_weight.blocks"
                    scale_name = f"block.0.mlp.{projection}_weight.scales"
                    shard = inspection.weights_root / index["weight_map"][block_name]
                    with safe_open(str(shard), framework="pt", device="cpu") as source:
                        blocks = source.get_slice(block_name)[0:1]
                        scales = source.get_slice(scale_name)[0:1]
                    reference = _convert_moe_packed_tensors(
                        blocks, scales, dtype=torch.bfloat16, rows_per_chunk=4096,
                    )[0].T.contiguous()
                    observed = staged.get_tensor(
                        f"block.0.mlp.{projection}_weight.expert.000"
                    )
                    self.assertTrue(torch.equal(
                        observed.view(torch.int16), reference.view(torch.int16)
                    ))

    @unittest.skipUnless(mxfp4.DEFAULT_MODEL_DIR.is_dir(), "local GPT-OSS source absent")
    def test_logical_parameter_manifest_is_generic_compatible_and_rejects_u8_count(self) -> None:
        inspection = mxfp4.inspect_model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "120B.json"
            manifest = mxfp4.build_logical_parameter_manifest(
                inspection, output_path=output
            )
            self.assertEqual([], mxfp4.validate_logical_parameter_manifest(
                manifest, verify_files=True
            ))
            self.assertEqual([], generic_parameter_manifest.validate_manifest(manifest))
            self.assertEqual(
                116_829_156_672,
                manifest["parameter_authority"]["exact_distinct_stored_parameter_count"],
            )
            damaged = json.loads(json.dumps(manifest))
            authority = damaged["parameter_authority"]
            authority["exact_distinct_stored_parameter_count"] = 63_081_444_672
            authority["stored_parameter_count"] = 63_081_444_672
            authority["dtype_element_counts"] = {"SERIALIZED_U8_AND_BF16": 63_081_444_672}
            authority["ownership_class_counts"] = {
                "serialized_elements": {"elements": 63_081_444_672, "tensors": 543}
            }
            damaged["manifest_sha256"] = mxfp4._hash_value({
                key: value for key, value in damaged.items()
                if key != "manifest_sha256"
            })
            errors = mxfp4.validate_logical_parameter_manifest(damaged)
            self.assertTrue(any("116829156672" in error for error in errors))

    @unittest.skipUnless(adapter.DEFAULT_INVENTORY.is_file(), "logical inventory absent")
    def test_adapter_contract_is_addressable_but_fail_closed(self) -> None:
        spec = adapter.build_spec(
            rate_id="0.5", cell_id="gptoss-120b-0p5-codec-test",
            cell_identity_sha256="1" * 64, program_spec_sha256="2" * 64,
            resource_admission_sha256="3" * 64,
            disk_reserve_bytes=150_000_000_000,
            scratch_budget_bytes=64_000_000_000, threads=8,
        )
        self.assertEqual([], adapter.validate_spec(spec))
        self.assertFalse(spec["execution_contract"]["codec_execution_executable"])
        self.assertFalse(spec["quality_claims_permitted"])
        self.assertEqual(116_829_156_672,
                         spec["logical_parameter_authority"]["logical_model_parameters"])
        self.assertEqual(5, len(spec["execution_contract"]["blocker_ids"]))


if __name__ == "__main__":
    unittest.main()
