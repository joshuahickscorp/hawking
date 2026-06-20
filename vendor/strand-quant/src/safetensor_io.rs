
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct StTensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub off_start: usize,
    pub off_end: usize,
}

pub struct SafeTensors {
    bytes: Vec<u8>,
    pub data_start: usize,
    pub tensors: HashMap<String, StTensor>,
    pub order: Vec<String>,
}

impl SafeTensors {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let json = bytes[8..8 + hdr_len].to_vec();
        let data_start = 8 + hdr_len;
        let (tensors, order) = parse_header(&json);
        Ok(SafeTensors { bytes, data_start, tensors, order })
    }

    pub fn raw(&self, t: &StTensor) -> &[u8] {
        &self.bytes[self.data_start + t.off_start..self.data_start + t.off_end]
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn to_f32(&self, t: &StTensor) -> Vec<f32> {
        let raw = self.raw(t);
        match t.dtype.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => panic!("unsupported safetensors dtype {other}"),
        }
    }
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = -1i32;
        let mut m = mant;
        loop {
            e += 1;
            m <<= 1;
            if m & 0x0400 != 0 {
                break;
            }
        }
        let mant32 = (m & 0x03ff) << 13;
        let exp32 = ((127 - 15 - e) as u32) << 23;
        f32::from_bits(sign | exp32 | mant32)
    } else if exp == 0x1f {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        let exp32 = ((exp as i32 - 15 + 127) as u32) << 23;
        let mant32 = mant << 13;
        f32::from_bits(sign | exp32 | mant32)
    }
}

pub fn parse_header(json: &[u8]) -> (HashMap<String, StTensor>, Vec<String>) {
    let s = std::str::from_utf8(json).expect("header is utf8");
    let b = s.as_bytes();
    let mut map = HashMap::new();
    let mut order = Vec::new();
    let mut i = 0usize;

    fn read_string(b: &[u8], mut i: usize) -> (String, usize) {
        debug_assert_eq!(b[i], b'"');
        i += 1;
        let start = i;
        while b[i] != b'"' {
            i += 1;
        }
        (String::from_utf8_lossy(&b[start..i]).into_owned(), i + 1)
    }

    while i < b.len() {
        if b[i] != b'"' {
            i += 1;
            continue;
        }
        let (key, ni) = read_string(b, i);
        i = ni;
        while i < b.len() && b[i] != b'{' && b[i] != b'"' && b[i] != b'[' {
            i += 1;
        }
        if key == "__metadata__" {
            let mut depth = 0;
            while i < b.len() {
                match b[i] {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            continue;
        }
        if i >= b.len() || b[i] != b'{' {
            continue;
        }
        let obj_start = i;
        let mut depth = 0;
        while i < b.len() {
            match b[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        let obj = &s[obj_start..i];
        let dtype = extract_str_field(obj, "dtype").unwrap_or_default();
        let shape = extract_num_array(obj, "shape");
        let offs = extract_num_array(obj, "data_offsets");
        if offs.len() == 2 {
            map.insert(
                key.clone(),
                StTensor { dtype, shape, off_start: offs[0] as usize, off_end: offs[1] as usize },
            );
            order.push(key);
        }
    }
    (map, order)
}

pub fn extract_str_field(obj: &str, field: &str) -> Option<String> {
    let pat = format!("\"{field}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = &obj[p..];
    let q1 = rest.find('"')?;
    let after = &rest[q1 + 1..];
    let q2 = after.find('"')?;
    Some(after[..q2].to_string())
}

pub fn extract_num_array(obj: &str, field: &str) -> Vec<u64> {
    let pat = format!("\"{field}\"");
    let Some(p) = obj.find(&pat) else { return Vec::new() };
    let rest = &obj[p + pat.len()..];
    let Some(lb) = rest.find('[') else { return Vec::new() };
    let Some(rb) = rest.find(']') else { return Vec::new() };
    rest[lb + 1..rb].split(',').filter_map(|t| t.trim().parse::<u64>().ok()).collect()
}
