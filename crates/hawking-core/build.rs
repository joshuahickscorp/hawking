fn main() {
    println!("cargo:rerun-if-changed=src/metal/physical_signpost.c");
    println!("cargo:rerun-if-changed=src/metal/ggml_fattn_authority.c");
    println!("cargo:rerun-if-changed=src/metal/capture_mps_gemm.m");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/metal/physical_signpost.c")
            .warnings_into_errors(true)
            .compile("hawking_physical_signpost");
        // This bridge is dynamically bound and is invoked only by the
        // explicit Llama K0 authority-adapter diagnostic. It does not add a
        // normal runtime dependency on a local ggml installation.
        cc::Build::new()
            .file("src/metal/ggml_fattn_authority.c")
            .warnings_into_errors(true)
            .compile("hawking_ggml_fattn_authority");
        // MPS GEMM bridge for Q30 layer-major activation capture (grouped/batched).
        cc::Build::new()
            .file("src/metal/capture_mps_gemm.m")
            .flag("-fobjc-arc")
            .warnings_into_errors(false)
            .compile("hawking_capture_mps_gemm");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
