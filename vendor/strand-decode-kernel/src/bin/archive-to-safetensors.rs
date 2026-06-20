
use std::io::Write;
use std::path::Path;

use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::outlier_mac::patched_weights;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [archive, out] = args.as_slice() else {
        eprintln!("usage: archive-to-safetensors <archive.strand> <out.safetensors>");
        std::process::exit(2);
    };

    let model = StrandModel::open(Path::new(archive)).expect("open .strand archive");
    let names: Vec<String> = model.tensor_names().map(|s| s.to_string()).collect();
    eprintln!("[a2s] {} tensors in {archive}", names.len());

    let mut header = String::from("{");
    let mut payload: Vec<u8> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let th = model.tensor_header(name).expect("tensor header");
        let shape: Vec<u64> = th.shape.clone();
        let w = patched_weights(&model, name)
            .unwrap_or_else(|e| panic!("patched decode {name}: {e}"));
        let start = payload.len();
        for v in &w {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();
        if i > 0 {
            header.push(',');
        }
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            dims.join(","),
            start,
            end
        ));
        if (i + 1) % 32 == 0 {
            eprintln!("[a2s] decoded {}/{}", i + 1, names.len());
        }
    }
    header.push('}');

    let mut f = std::fs::File::create(out).expect("create out safetensors");
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.sync_all().ok();
    eprintln!(
        "[a2s] wrote {out}: {} tensors, {} payload bytes (F32, patched decode)",
        names.len(),
        payload.len()
    );
}
