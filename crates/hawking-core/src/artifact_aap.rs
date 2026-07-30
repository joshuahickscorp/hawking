//! AAP v1 activation-aware.f16 codec (header/tensor public surface).
use crate::{Error, Result};
use half::f16;
const ACTIVATION_AWARE_MAGIC: &[u8; 8] = b"GLM52AAP";
const ACTIVATION_AWARE_HEADER_LEN: usize = 64;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationAwareSide {
    Input,
    Output,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationAwareHeader {
    pub rows: u32,
    pub cols: u32,
    pub rank: u32,
    pub basis_layer: u16,
    pub side: ActivationAwareSide,
    pub has_basis: bool,
}
pub fn parse_activation_aware_header(payload: &[u8]) -> Result<ActivationAwareHeader> {
    if payload.len() < ACTIVATION_AWARE_HEADER_LEN {
        return Err(Error::Gravity(format!(
            "aap too short: {} bytes, need {ACTIVATION_AWARE_HEADER_LEN}",
            payload.len()
        )));
    }
    require_magic!(payload, ACTIVATION_AWARE_MAGIC, "activation-aware")?;
    let rows = le_u32!(payload, 8);
    let cols = le_u32!(payload, 12);
    let rank = le_u32!(payload, 16);
    let basis_layer = le_u16!(payload, 20);
    let side = match le_u16!(payload, 22) {
        1 => ActivationAwareSide::Input,
        2 => ActivationAwareSide::Output,
        other => return Err(Error::Gravity(format!("aap side {other}"))),
    };
    let has_basis = match payload[24] {
        0 | 1 => payload[24] == 1,
        o => return Err(Error::Gravity(format!("aap has_basis {o} is not 0/1"))),
    };
    if rows == 0 || cols == 0 || rank == 0 {
        return Err(Error::Gravity(format!(
            "aap geom rows={rows}, cols={cols}, rank={rank}"
        )));
    }
    let side_width = match side {
        ActivationAwareSide::Input => cols,
        ActivationAwareSide::Output => rows,
    };
    if rank > side_width {
        return Err(Error::Gravity(format!(
            "activation-aware rank {rank} exceeds basis width {side_width}"
        )));
    }
    Ok(ActivationAwareHeader {
        rows,
        cols,
        rank,
        basis_layer,
        side,
        has_basis,
    })
}
pub fn activation_aware_sections(payload: &[u8]) -> Result<(ActivationAwareHeader, &[u8], &[u8])> {
    let header = parse_activation_aware_header(payload)?;
    if !header.has_basis {
        return Err(Error::Gravity("aap no basis".into()));
    }
    let rows = header.rows as usize;
    let cols = header.cols as usize;
    let rank = header.rank as usize;
    let coefficient_values = match header.side {
        ActivationAwareSide::Input => rows.checked_mul(rank),
        ActivationAwareSide::Output => rank.checked_mul(cols),
    }
    .ok_or_else(|| Error::Gravity("aap coeff overflow".into()))?;
    let basis_values = match header.side {
        ActivationAwareSide::Input => cols.checked_mul(rank),
        ActivationAwareSide::Output => rows.checked_mul(rank),
    }
    .ok_or_else(|| Error::Gravity("aap basis overflow".into()))?;
    let coefficient_end = section_end!(
        ACTIVATION_AWARE_HEADER_LEN,
        coefficient_values,
        2,
        "aap-coeff"
    );
    let expected = section_end!(coefficient_end, basis_values, 2, "aap-payload");
    if payload.len() != expected {
        return Err(Error::Gravity(format!(
            "aap bytes {} != {expected}",
            payload.len()
        )));
    }
    Ok((
        header,
        &payload[ACTIVATION_AWARE_HEADER_LEN..coefficient_end],
        &payload[coefficient_end..],
    ))
}
pub struct ActivationAwareTensor {
    pub header: ActivationAwareHeader,
    coefficients: Vec<f32>,
    basis: Vec<f32>,
}
impl ActivationAwareTensor {
    pub fn from_payload(payload: &[u8]) -> Result<Self> {
        let (header, coefficient_bytes, basis_bytes) = activation_aware_sections(payload)?;
        Ok(Self {
            header,
            coefficients: (|bytes: &[u8]| {
                bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect::<Vec<f32>>()
            })(coefficient_bytes),
            basis: (|bytes: &[u8]| {
                bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect::<Vec<f32>>()
            })(basis_bytes),
        })
    }
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        let rows = self.header.rows as usize;
        let cols = self.header.cols as usize;
        let rank = self.header.rank as usize;
        if x.len() != cols {
            return Err(Error::Gravity(format!("aap matvec {} != {cols}", x.len())));
        }
        let mut latent = vec![0.0f32; rank];
        let mut out = vec![0.0f32; rows];
        match self.header.side {
            ActivationAwareSide::Input => {
                for (col, &value) in x.iter().enumerate() {
                    let basis_row = &self.basis[col * rank..(col + 1) * rank];
                    for k in 0..rank {
                        latent[k] += basis_row[k] * value;
                    }
                }
                for (row, target) in out.iter_mut().enumerate() {
                    let coefficient_row = &self.coefficients[row * rank..(row + 1) * rank];
                    *target = coefficient_row
                        .iter()
                        .zip(&latent)
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
            ActivationAwareSide::Output => {
                for (k, target) in latent.iter_mut().enumerate() {
                    let coefficient_row = &self.coefficients[k * cols..(k + 1) * cols];
                    *target = coefficient_row.iter().zip(x).map(|(a, b)| a * b).sum();
                }
                for (row, target) in out.iter_mut().enumerate() {
                    let basis_row = &self.basis[row * rank..(row + 1) * rank];
                    *target = basis_row.iter().zip(&latent).map(|(a, b)| a * b).sum();
                }
            }
        }
        Ok(out)
    }
    pub fn row(&self, index: usize) -> Result<Vec<f32>> {
        let rows = self.header.rows as usize;
        let cols = self.header.cols as usize;
        let rank = self.header.rank as usize;
        if index >= rows {
            return Err(Error::Gravity(format!(
                "activation-aware row {index} out of range {rows}"
            )));
        }
        let mut out = vec![0.0f32; cols];
        match self.header.side {
            ActivationAwareSide::Input => {
                let coefficient_row = &self.coefficients[index * rank..(index + 1) * rank];
                for (col, target) in out.iter_mut().enumerate() {
                    let basis_row = &self.basis[col * rank..(col + 1) * rank];
                    *target = coefficient_row
                        .iter()
                        .zip(basis_row)
                        .map(|(a, b)| a * b)
                        .sum();
                }
            }
            ActivationAwareSide::Output => {
                let basis_row = &self.basis[index * rank..(index + 1) * rank];
                for col in 0..cols {
                    for (k, &basis_value) in basis_row.iter().enumerate() {
                        out[col] += basis_value * self.coefficients[k * cols + col];
                    }
                }
            }
        }
        Ok(out)
    }
}
