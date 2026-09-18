#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BlockParams {
    pub(crate) block_len: u32,
    pub(crate) num_states: u32,
    pub(crate) k_bits: u32,
    pub(crate) n_sub: u32,
    pub(crate) sub_size: u32,
    pub(crate) max_block_len: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}

pub struct GpuViterbiResult {
    pub back_flat: Vec<u32>,

    pub final_cost: Vec<f32>,

    pub max_block_len: usize,
}
