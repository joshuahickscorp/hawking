fn main() {
    println!("cargo:rerun-if-changed=src/metal_physical_signpost.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/metal_physical_signpost.c")
            .warnings_into_errors(true)
            .compile("hawking_physical_signpost");
    }
}
