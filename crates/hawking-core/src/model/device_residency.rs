//! Artifact-static payload residency and one-shot address-table indirection.
//!
//! Law this module encodes:
//!
//! ```text
//! artifact-static work  -> admission (paid once)
//! session-static work   -> setup
//! token-dynamic work    -> write selected route ids; kernel indirects
//! ```
//!
//! The mechanism is payload-layout parameterized. Uniform-Q4, binary_group,
//! binary+rice residual, and low-rank factors all drop in by implementing
//! [`ResidentPayload`] and filling [`PayloadLayout`]. The Q4 catalog is a
//! test harness, not a hard-coded representation.
//!
//! Token loop contract:
//!
//! ```text
//! ADMISSION: payloads resident and device-addressable;
//!            ONE address table covering n_layers * n_experts entries
//! TOKEN:     write the selected route ids (tens of bytes)
//!            kernel: addr_table[layer * n_experts + route_id[k]]
//!                    (implemented as a buffer bind offset of
//!                     layer * n_experts * entry_bytes, so existing
//!                     kernels that index table[route_id] keep working)
//! ```

use crate::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Session-static generation written into every admitted table slot.
/// Token kernels compare the bound generation against this value.
pub const ADDRESS_TABLE_GENERATION: u32 = 1;

/// Default expert-payload budget. Full Q4 residency (~38 GiB of expert
/// bodies) is larger than the existing 16 GiB streamed RSS cap, so the
/// pool holds a resident subset and LRU-evicts the tail. The 12-token
/// Q4 harness touches ~4383 unique experts (~6.82 GiB); 8 GiB keeps
/// that working set resident. The <=1.5 artifact shrinks the body and
/// inherits the same pool.
pub const DEFAULT_RESIDENCY_BUDGET_MIB: usize = 8192;

/// Bytes of one address-table entry in the current 40+40+40+8 triplet ABI
/// (three tensor refs + ready_mask + generation). New payload families
/// that need a different entry width set [`PayloadLayout::entry_bytes`].
pub const TRIPLET_ENTRY_BYTES: usize = 128;

/// How one resident payload is described in the address table.
///
/// `kind` is the family tag the consuming kernel understands
/// (`UNIFORM_Q4 = 3` on the Q4 harness). `entry_bytes` is the table-slot
/// width. `resource_count` is how many device buffers one payload keeps
/// alive (six for Q4 gate/up/down × codes/scales).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayloadLayout {
    pub kind: u32,
    pub entry_bytes: usize,
    pub resource_count: usize,
    pub payload_bytes: u64,
}

impl PayloadLayout {
    pub const fn triplet(kind: u32, resource_count: usize, payload_bytes: u64) -> Self {
        Self {
            kind,
            entry_bytes: TRIPLET_ENTRY_BYTES,
            resource_count,
            payload_bytes,
        }
    }
}

/// One static table covering every `(layer, expert)` slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressTableGeometry {
    pub n_layers: usize,
    pub n_experts: usize,
    pub layout: PayloadLayout,
}

impl AddressTableGeometry {
    pub fn table_bytes(&self) -> usize {
        self.n_layers
            .saturating_mul(self.n_experts)
            .saturating_mul(self.layout.entry_bytes)
    }

    pub fn slot_index(&self, layer: usize, expert: usize) -> Result<usize> {
        if layer >= self.n_layers {
            return Err(residency_error(format!(
                "layer {layer} is outside 0..{}",
                self.n_layers
            )));
        }
        if expert >= self.n_experts {
            return Err(residency_error(format!(
                "expert {expert} is outside 0..{}",
                self.n_experts
            )));
        }
        Ok(layer
            .saturating_mul(self.n_experts)
            .saturating_add(expert))
    }

    pub fn slot_byte_offset(&self, layer: usize, expert: usize) -> Result<usize> {
        Ok(self
            .slot_index(layer, expert)?
            .saturating_mul(self.layout.entry_bytes))
    }

    pub fn layer_byte_offset(&self, layer: usize) -> Result<usize> {
        if layer >= self.n_layers {
            return Err(residency_error(format!(
                "layer {layer} is outside 0..{}",
                self.n_layers
            )));
        }
        Ok(layer
            .saturating_mul(self.n_experts)
            .saturating_mul(self.layout.entry_bytes))
    }

    pub fn full_payload_bytes(&self) -> u64 {
        (self.n_layers as u64)
            .saturating_mul(self.n_experts as u64)
            .saturating_mul(self.layout.payload_bytes)
    }

    /// Linear scale of the current payload by `target_bpw / current_bpw`.
    /// Used only to report the easier <=1.5 footprint; it is not a pack.
    pub fn scaled_payload_bytes(&self, current_bpw: f64, target_bpw: f64) -> u64 {
        if current_bpw <= 0.0 {
            return self.full_payload_bytes();
        }
        let scaled = (self.full_payload_bytes() as f64) * (target_bpw / current_bpw);
        if scaled.is_finite() && scaled > 0.0 {
            scaled.round() as u64
        } else {
            self.full_payload_bytes()
        }
    }
}

/// Default **on**. Set `HAWKING_PERSISTENT_ADDRESS_TABLE=0` (or the
/// Q80-scoped alias) to restore the per-layer table rewrite for A/B.
pub fn persistent_address_table_enabled() -> bool {
    crate::env_opt_out("HAWKING_PERSISTENT_ADDRESS_TABLE")
        && crate::env_opt_out("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE")
}

/// Resident-payload budget in bytes. Override with `HAWKING_RESIDENCY_BUDGET_MIB`.
pub fn residency_budget_bytes() -> u64 {
    let mib = crate::env_usize("HAWKING_RESIDENCY_BUDGET_MIB", DEFAULT_RESIDENCY_BUDGET_MIB) as u64;
    mib.saturating_mul(1024 * 1024)
}

#[derive(Clone, Debug, Default)]
pub struct ResidencyStats {
    pub upload_hits: u64,
    pub upload_misses: u64,
    pub evictions: u64,
    pub resident_slots: u64,
    pub resident_bytes: u64,
    pub table_slot_patches: u64,
    pub upload_miss_secs: f64,
    pub entries_fill_secs: f64,
    pub buffer_write_secs: f64,
    pub resource_clone_secs: f64,
    pub lease_secs: f64,
}

impl ResidencyStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.upload_hits.saturating_add(self.upload_misses);
        if total == 0 {
            0.0
        } else {
            self.upload_hits as f64 / total as f64
        }
    }
}

fn residency_error(message: impl Into<String>) -> Error {
    Error::Model(format!("device residency: {}", message.into()))
}

#[cfg(target_os = "macos")]
pub trait ResidentPayload {
    fn payload_bytes(&self) -> u64;
    fn write_entry(&self, generation: u32, dst: &mut [u8]);
    fn append_resources(&self, dst: &mut Vec<crate::metal::PinnedBuffer>);
}

#[cfg(target_os = "macos")]
struct ResidentSlot<P> {
    payload: P,
    last_tick: u64,
}

/// Bind record for one layer's selected experts. The table handle is the
/// all-layer buffer; `table_byte_offset` is `layer * n_experts * entry_bytes`
/// so a kernel that indexes `table[route_id]` sees that layer's 512-way slice.
#[cfg(target_os = "macos")]
pub struct LayerBind {
    pub table: crate::metal::PinnedBuffer,
    pub table_byte_offset: u64,
    pub resources: Vec<crate::metal::PinnedBuffer>,
    pub generation: u32,
    pub n_experts: usize,
    pub layer: usize,
}

#[cfg(target_os = "macos")]
pub struct PersistentAddressTable {
    pub geometry: AddressTableGeometry,
    table: crate::metal::PinnedBuffer,
    generation: u32,
    entry_scratch: Vec<u8>,
}

#[cfg(target_os = "macos")]
impl PersistentAddressTable {
    pub fn allocate(
        context: &crate::metal::MetalContext,
        geometry: AddressTableGeometry,
    ) -> Result<Self> {
        if geometry.layout.entry_bytes == 0 {
            return Err(residency_error("payload layout entry_bytes is zero"));
        }
        let bytes = geometry.table_bytes();
        if bytes == 0 {
            return Err(residency_error("address table geometry is empty"));
        }
        Ok(Self {
            geometry,
            table: context.new_buffer_checked(bytes)?,
            generation: ADDRESS_TABLE_GENERATION,
            entry_scratch: vec![0u8; geometry.layout.entry_bytes],
        })
    }

    pub fn table(&self) -> &crate::metal::PinnedBuffer {
        &self.table
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn write_slot<P: ResidentPayload>(&mut self, layer: usize, expert: u32, payload: &P) -> Result<()> {
        let offset = self.geometry.slot_byte_offset(layer, expert as usize)?;
        let entry_bytes = self.geometry.layout.entry_bytes;
        if self.entry_scratch.len() != entry_bytes {
            self.entry_scratch.resize(entry_bytes, 0);
        }
        self.entry_scratch.fill(0);
        payload.write_entry(self.generation, &mut self.entry_scratch);
        write_table_bytes(&self.table, offset, &self.entry_scratch);
        Ok(())
    }

    pub fn clear_slot(&mut self, layer: usize, expert: u32) -> Result<()> {
        let offset = self.geometry.slot_byte_offset(layer, expert as usize)?;
        let zeros = vec![0u8; self.geometry.layout.entry_bytes];
        write_table_bytes(&self.table, offset, &zeros);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_table_bytes(buf: &crate::metal::PinnedBuffer, offset: usize, bytes: &[u8]) {
    let len = buf.length() as usize;
    debug_assert!(offset.saturating_add(bytes.len()) <= len);
    unsafe {
        let dst = (buf.contents() as *mut u8).add(offset);
        dst.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
    }
}

/// LRU resident-payload pool over one persistent all-layer address table.
#[cfg(target_os = "macos")]
pub struct ResidencyPool<P> {
    table: PersistentAddressTable,
    slots: HashMap<(usize, u32), ResidentSlot<P>>,
    lru: VecDeque<(usize, u32, u64)>,
    tick: u64,
    budget_bytes: u64,
    resident_bytes: u64,
    pub stats: ResidencyStats,
}

#[cfg(target_os = "macos")]
impl<P: ResidentPayload> ResidencyPool<P> {
    pub fn new(table: PersistentAddressTable, budget_bytes: u64) -> Self {
        Self {
            table,
            slots: HashMap::new(),
            lru: VecDeque::new(),
            tick: 0,
            budget_bytes,
            resident_bytes: 0,
            stats: ResidencyStats::default(),
        }
    }

    pub fn geometry(&self) -> AddressTableGeometry {
        self.table.geometry
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn contains(&self, layer: usize, expert: u32) -> bool {
        self.slots.contains_key(&(layer, expert))
    }

    pub fn ensure_selected<F>(
        &mut self,
        layer: usize,
        experts: &[u32],
        mut upload: F,
    ) -> Result<()>
    where
        F: FnMut(usize, u32) -> Result<P>,
    {
        let mut pinned: Vec<(usize, u32)> = Vec::with_capacity(experts.len());
        for &expert in experts {
            let _ = self.table.geometry.slot_index(layer, expert as usize)?;
            if self.slots.contains_key(&(layer, expert)) {
                self.stats.upload_hits = self.stats.upload_hits.saturating_add(1);
                self.touch(layer, expert);
                pinned.push((layer, expert));
                continue;
            }
            self.stats.upload_misses = self.stats.upload_misses.saturating_add(1);
            let hint = self.table.geometry.layout.payload_bytes;
            self.evict_for(hint, &pinned)?;
            let upload_started = Instant::now();
            let payload = upload(layer, expert)?;
            self.stats.upload_miss_secs += upload_started.elapsed().as_secs_f64();
            let actual = payload.payload_bytes();
            if actual > hint {
                self.evict_for(actual, &pinned)?;
            }
            self.insert_resident(layer, expert, payload)?;
            pinned.push((layer, expert));
        }
        Ok(())
    }

    pub fn layer_bind(&mut self, layer: usize, experts: &[u32]) -> Result<LayerBind> {
        let offset = self.table.geometry.layer_byte_offset(layer)? as u64;
        let clone_started = Instant::now();
        let mut resources = Vec::with_capacity(
            experts
                .len()
                .saturating_mul(self.table.geometry.layout.resource_count),
        );
        for &expert in experts {
            let slot = self.slots.get(&(layer, expert)).ok_or_else(|| {
                residency_error(format!(
                    "layer {layer} expert {expert} is not resident at bind"
                ))
            })?;
            slot.payload.append_resources(&mut resources);
        }
        self.stats.resource_clone_secs += clone_started.elapsed().as_secs_f64();
        let lease_started = Instant::now();
        let bind = LayerBind {
            table: self.table.table().clone(),
            table_byte_offset: offset,
            resources,
            generation: self.table.generation(),
            n_experts: self.table.geometry.n_experts,
            layer,
        };
        self.stats.lease_secs += lease_started.elapsed().as_secs_f64();
        Ok(bind)
    }

    fn insert_resident(&mut self, layer: usize, expert: u32, payload: P) -> Result<()> {
        let fill_started = Instant::now();
        let bytes = payload.payload_bytes();
        let write_started = Instant::now();
        self.table.write_slot(layer, expert, &payload)?;
        self.stats.buffer_write_secs += write_started.elapsed().as_secs_f64();
        self.stats.entries_fill_secs += fill_started.elapsed().as_secs_f64();
        self.stats.table_slot_patches = self.stats.table_slot_patches.saturating_add(1);
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        self.tick = self.tick.saturating_add(1);
        self.slots.insert(
            (layer, expert),
            ResidentSlot {
                payload,
                last_tick: self.tick,
            },
        );
        self.lru.push_back((layer, expert, self.tick));
        self.stats.resident_slots = self.slots.len() as u64;
        self.stats.resident_bytes = self.resident_bytes;
        Ok(())
    }

    fn touch(&mut self, layer: usize, expert: u32) {
        self.tick = self.tick.saturating_add(1);
        if let Some(slot) = self.slots.get_mut(&(layer, expert)) {
            slot.last_tick = self.tick;
            self.lru.push_back((layer, expert, self.tick));
        }
    }

    fn evict_for(&mut self, needed: u64, pinned: &[(usize, u32)]) -> Result<()> {
        while self.resident_bytes.saturating_add(needed) > self.budget_bytes {
            let Some((layer, expert, tick)) = self.lru.pop_front() else {
                return Err(residency_error(format!(
                    "budget {} B cannot admit {needed} B; resident {} B and LRU is empty",
                    self.budget_bytes, self.resident_bytes
                )));
            };
            let Some(slot) = self.slots.get(&(layer, expert)) else {
                continue;
            };
            if slot.last_tick != tick {
                continue;
            }
            if pinned.iter().any(|&key| key == (layer, expert)) {
                self.lru.push_back((layer, expert, tick));
                if self.lru.len() == 1 {
                    return Err(residency_error(format!(
                        "budget {} B cannot admit {needed} B without evicting a live route",
                        self.budget_bytes
                    )));
                }
                continue;
            }
            let bytes = slot.payload.payload_bytes();
            self.table.clear_slot(layer, expert)?;
            self.slots.remove(&(layer, expert));
            self.resident_bytes = self.resident_bytes.saturating_sub(bytes);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            self.stats.resident_slots = self.slots.len() as u64;
            self.stats.resident_bytes = self.resident_bytes;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q4_harness_geometry() -> AddressTableGeometry {
        AddressTableGeometry {
            n_layers: 48,
            n_experts: 512,
            layout: PayloadLayout::triplet(3, 6, 1_671_168),
        }
    }

    #[test]
    fn all_layer_table_is_three_mib_not_per_token() {
        let geometry = q4_harness_geometry();
        assert_eq!(geometry.table_bytes(), 48 * 512 * 128);
        assert_eq!(geometry.table_bytes(), 3_145_728);
        assert_eq!(geometry.slot_index(0, 0).unwrap(), 0);
        assert_eq!(geometry.slot_index(1, 0).unwrap(), 512);
        assert_eq!(geometry.slot_byte_offset(2, 7).unwrap(), (2 * 512 + 7) * 128);
        assert_eq!(geometry.layer_byte_offset(7).unwrap(), 7 * 512 * 128);
        assert!(geometry.slot_index(48, 0).is_err());
        assert!(geometry.slot_index(0, 512).is_err());
    }

    #[test]
    fn q4_full_residency_exceeds_16gib_cap_and_1_5_is_smaller() {
        let geometry = q4_harness_geometry();
        let full = geometry.full_payload_bytes();
        assert_eq!(full, 48 * 512 * 1_671_168);
        assert_eq!(full, 41_070_624_768);
        let cap_16 = 16u64 * 1024 * 1024 * 1024;
        assert!(
            full > cap_16,
            "Q4 full expert residency must remain blocked by the existing 16 GiB RSS cap"
        );
        let at_1_5 = geometry.scaled_payload_bytes(4.259241, 1.5);
        assert!(at_1_5 < full);
        assert!(
            at_1_5 < cap_16,
            "<=1.5 scaled expert bodies should fit the existing streamed cap"
        );
    }

    #[test]
    fn persistent_table_defaults_on() {
        let previous = std::env::var("HAWKING_PERSISTENT_ADDRESS_TABLE").ok();
        let previous_q80 = std::env::var("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE").ok();
        std::env::remove_var("HAWKING_PERSISTENT_ADDRESS_TABLE");
        std::env::remove_var("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE");
        assert!(persistent_address_table_enabled());
        std::env::set_var("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE", "0");
        assert!(!persistent_address_table_enabled());
        std::env::set_var("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE", "1");
        assert!(persistent_address_table_enabled());
        match previous {
            Some(value) => std::env::set_var("HAWKING_PERSISTENT_ADDRESS_TABLE", value),
            None => std::env::remove_var("HAWKING_PERSISTENT_ADDRESS_TABLE"),
        }
        match previous_q80 {
            Some(value) => std::env::set_var("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE", value),
            None => std::env::remove_var("HAWKING_QWEN80_PERSISTENT_ADDRESS_TABLE"),
        }
    }

    #[test]
    fn hit_rate_is_zero_until_traffic() {
        let stats = ResidencyStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
        let stats = ResidencyStats {
            upload_hits: 9,
            upload_misses: 1,
            ..ResidencyStats::default()
        };
        assert!((stats.hit_rate() - 0.9).abs() < 1e-12);
    }
}
