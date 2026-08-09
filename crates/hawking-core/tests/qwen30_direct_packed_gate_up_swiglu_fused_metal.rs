#![cfg(target_os = "macos")]

//! Fixture-level parity gate for the all-row Qwen30 direct-packed gate/up
//! SwiGLU candidate.  This does not admit a model artifact and cannot make a
//! model/TPS claim; it protects the fixed HQ30G1B1 byte layout and the fused
//! component's numerical semantics.

use half::f16;
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

const ROWS: usize = 768;
const COLS: usize = 2048;
const GROUP: usize = 128;
const GROUPS_PER_ROW: usize = COLS / GROUP;
const TOLERANCE: f32 = 4.0e-3;
const SHADER_SOURCE: &str =
    include_str!("../shaders/qwen_direct_packed_gate_up_swiglu_fused.metal");

fn make_signs(seed: usize) -> Vec<u8> {
    let mut signs = vec![0u8; ROWS * COLS / 8];
    for element in 0..ROWS * COLS {
        let value = element
            .wrapping_mul(73)
            .wrapping_add(seed.wrapping_mul(19))
            .wrapping_add(element / COLS * 31);
        if value % 11 >= 5 {
            signs[element >> 3] |= 1 << (element & 7);
        }
    }
    signs
}

fn make_scales(seed: usize) -> Vec<u16> {
    (0..ROWS * GROUPS_PER_ROW)
        .map(|group| {
            let bucket = ((group * 29 + seed * 17) % 97) as f32;
            f16::from_f32(0.0025 + bucket / 4096.0).to_bits()
        })
        .collect()
}

fn make_input() -> Vec<f32> {
    (0..COLS)
        .map(|column| ((column * 71 % 509) as f32 - 254.0) / 509.0)
        .collect()
}

fn direct_value(signs: &[u8], scales: &[u16], element: usize) -> f32 {
    let group = element / GROUP;
    let sign = ((signs[element >> 3] >> (element & 7)) & 1) != 0;
    let scale = f16::from_bits(scales[group]).to_f32();
    if sign {
        scale
    } else {
        -scale
    }
}

fn cpu_matvec(signs: &[u8], scales: &[u16], input: &[f32]) -> Vec<f32> {
    (0..ROWS)
        .map(|row| {
            let mut sum = 0.0f32;
            for column in 0..COLS {
                sum = direct_value(signs, scales, row * COLS + column).mul_add(input[column], sum);
            }
            sum
        })
        .collect()
}

fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

fn max_error(expected: &[f32], actual: &[f32]) -> f32 {
    expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &value as *const u32 as *const _,
    );
}

fn byte_buffer(device: &Device, bytes: &[u8]) -> metal::Buffer {
    let buffer = device.new_buffer(bytes.len() as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents() as *mut u8, bytes.len())
    };
    buffer
}

fn f32_buffer(device: &Device, values: &[f32]) -> metal::Buffer {
    let buffer = device.new_buffer(
        (values.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            buffer.contents() as *mut u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    buffer
}

fn complete(command: &metal::CommandBufferRef) {
    command.commit();
    command.wait_until_completed();
    assert_eq!(
        command.status(),
        MTLCommandBufferStatus::Completed,
        "Metal component command did not complete"
    );
}

fn read_f32(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, len).to_vec() }
}

#[test]
fn all_row_direct_packed_gate_up_swiglu_fused_matches_cpu_and_control_path() {
    let device = match Device::system_default() {
        Some(device) if device.has_unified_memory() => device,
        Some(_) => {
            eprintln!(
                "skipping: direct-packed fixture requires unified-memory Apple Silicon Metal"
            );
            return;
        }
        None => {
            eprintln!("skipping: Metal device unavailable");
            return;
        }
    };
    assert_eq!(ROWS, 768);
    assert_eq!(COLS, 2048);
    assert_eq!(GROUPS_PER_ROW, 16);
    assert_eq!(ROWS * COLS / 8, 196_608);
    assert_eq!(ROWS * GROUPS_PER_ROW * std::mem::size_of::<u16>(), 24_576);

    let gate_signs = make_signs(1);
    let gate_scales = make_scales(1);
    let up_signs = make_signs(7);
    let up_scales = make_scales(7);
    let input = make_input();
    let gate_cpu = cpu_matvec(&gate_signs, &gate_scales, &input);
    let up_cpu = cpu_matvec(&up_signs, &up_scales, &input);
    let activation_cpu = swiglu(&gate_cpu, &up_cpu);

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(SHADER_SOURCE, &options)
        .expect("compile fused direct-packed component shader");
    let matvec = device
        .new_compute_pipeline_state_with_function(
            &library
                .get_function("qwen_direct_packed_gate_up_baseline_matvec", None)
                .expect("baseline matvec function"),
        )
        .expect("baseline matvec pipeline");
    let baseline_swiglu = device
        .new_compute_pipeline_state_with_function(
            &library
                .get_function("qwen_direct_packed_gate_up_baseline_swiglu", None)
                .expect("baseline SwiGLU function"),
        )
        .expect("baseline SwiGLU pipeline");
    let fused = device
        .new_compute_pipeline_state_with_function(
            &library
                .get_function("qwen_direct_packed_gate_up_swiglu_fused_candidate", None)
                .expect("fused candidate function"),
        )
        .expect("fused candidate pipeline");

    let gate_signs_buffer = byte_buffer(&device, &gate_signs);
    let gate_scales_buffer = byte_buffer(&device, bytemuck::cast_slice(&gate_scales));
    let up_signs_buffer = byte_buffer(&device, &up_signs);
    let up_scales_buffer = byte_buffer(&device, bytemuck::cast_slice(&up_scales));
    let input_buffer = f32_buffer(&device, &input);
    let gate_output = f32_buffer(&device, &vec![0.0; ROWS]);
    let up_output = f32_buffer(&device, &vec![0.0; ROWS]);
    let baseline_activation = f32_buffer(&device, &vec![0.0; ROWS]);
    let fused_activation = f32_buffer(&device, &vec![0.0; ROWS]);
    let queue = device.new_command_queue();

    let baseline_command = queue.new_command_buffer();
    for (signs, scales, output) in [
        (&gate_signs_buffer, &gate_scales_buffer, &gate_output),
        (&up_signs_buffer, &up_scales_buffer, &up_output),
    ] {
        let encoder = baseline_command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&matvec);
        encoder.set_buffer(0, Some(signs), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(output), 0);
        set_u32(encoder, 4, ROWS as u32);
        set_u32(encoder, 5, COLS as u32);
        set_u32(encoder, 6, GROUP as u32);
        encoder.dispatch_threads(MTLSize::new(ROWS as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }
    let encoder = baseline_command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&baseline_swiglu);
    encoder.set_buffer(0, Some(&gate_output), 0);
    encoder.set_buffer(1, Some(&up_output), 0);
    encoder.set_buffer(2, Some(&baseline_activation), 0);
    set_u32(encoder, 3, ROWS as u32);
    encoder.dispatch_threads(MTLSize::new(ROWS as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    complete(baseline_command);

    let fused_command = queue.new_command_buffer();
    let encoder = fused_command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&fused);
    encoder.set_buffer(0, Some(&gate_signs_buffer), 0);
    encoder.set_buffer(1, Some(&gate_scales_buffer), 0);
    encoder.set_buffer(2, Some(&up_signs_buffer), 0);
    encoder.set_buffer(3, Some(&up_scales_buffer), 0);
    encoder.set_buffer(4, Some(&input_buffer), 0);
    encoder.set_buffer(5, Some(&fused_activation), 0);
    set_u32(encoder, 6, ROWS as u32);
    set_u32(encoder, 7, COLS as u32);
    set_u32(encoder, 8, GROUP as u32);
    encoder.dispatch_threads(MTLSize::new(ROWS as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    complete(fused_command);

    let gate_metal = read_f32(&gate_output, ROWS);
    let up_metal = read_f32(&up_output, ROWS);
    let baseline_metal = read_f32(&baseline_activation, ROWS);
    let fused_metal = read_f32(&fused_activation, ROWS);
    let gate_error = max_error(&gate_cpu, &gate_metal);
    let up_error = max_error(&up_cpu, &up_metal);
    let baseline_error = max_error(&activation_cpu, &baseline_metal);
    let fused_error = max_error(&activation_cpu, &fused_metal);
    let control_delta = max_error(&baseline_metal, &fused_metal);
    assert!(
        gate_error <= TOLERANCE
            && up_error <= TOLERANCE
            && baseline_error <= TOLERANCE
            && fused_error <= TOLERANCE
            && control_delta <= TOLERANCE,
        "all-row direct-packed fused parity failed: gate={gate_error}, up={up_error}, baseline={baseline_error}, fused={fused_error}, delta={control_delta}, tolerance={TOLERANCE}"
    );
}
