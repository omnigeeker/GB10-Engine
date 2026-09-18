//! CPU reference decoders for the quantized formats.
//!
//! Used to prove the GPU kernels right, independently of any GPU code.

/// FP4 E2M1: 1 sign, 2 exponent, 1 mantissa; magnitudes 0,.5,1,1.5,2,3,4,6.
pub fn e2m1_to_f32(nibble: u8) -> f32 {
    let e = (nibble >> 1) & 0x3;
    let m = nibble & 0x1;
    let mag = if e == 0 {
        if m == 1 {
            0.5
        } else {
            0.0
        }
    } else {
        (if m == 1 { 1.5 } else { 1.0 }) * 2f32.powi(e as i32 - 1)
    };
    if nibble & 0x8 != 0 {
        -mag
    } else {
        mag
    }
}

/// FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa, subnormals at exp 0.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    let mag = if e == 0 {
        m * (1.0 / 8.0) * 2f32.powi(-6)
    } else if e == 0xF && (b & 0x7) == 0x7 {
        f32::NAN
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    sign * mag
}

/// bf16 is the top half of an fp32.
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Dequantize one NVFP4 row of `k` elements.
///
/// `packed` is `k/2` bytes (low nibble first), `scales` is `k/16` E4M3 bytes,
/// `scale2` is the per-tensor global scale.
pub fn dequant_nvfp4_row(packed: &[u8], scales: &[u8], scale2: f32, k: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(k);
    for e in 0..k {
        let byte = packed[e >> 1];
        let nib = if e & 1 == 1 { byte >> 4 } else { byte & 0xF };
        let sc = e4m3_to_f32(scales[e >> 4]);
        out.push(e2m1_to_f32(nib) * sc * scale2);
    }
    out
}

/// Reference NVFP4 GEMV on the host.
pub fn nvfp4_gemv_ref(
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    scale2: f32,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    let rowbytes = k / 2;
    let scalerow = k / 16;
    for row in 0..n {
        let w = dequant_nvfp4_row(
            &packed[row * rowbytes..(row + 1) * rowbytes],
            &scales[row * scalerow..(row + 1) * scalerow],
            scale2,
            k,
        );
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += w[i] * x[i];
        }
        y[row] = acc;
    }
    y
}

/// Reference FP8 GEMV on the host.
pub fn fp8_gemv_ref(x: &[f32], w: &[u8], wscale: f32, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += e4m3_to_f32(w[row * k + i]) * x[i];
        }
        y[row] = acc * wscale;
    }
    y
}

/// Reference bf16 GEMV on the host.
pub fn bf16_gemv_ref(x: &[f32], w: &[u16], n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for i in 0..k {
            acc += bf16_to_f32(w[row * k + i]) * x[i];
        }
        y[row] = acc;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_matches_the_spec_table() {
        // nibble: (e,m) -> magnitude
        let cases = [
            (0b0000u8, 0.0f32),
            (0b0001, 0.5),
            (0b0010, 1.0),
            (0b0011, 1.5),
            (0b0100, 2.0),
            (0b0101, 3.0),
            (0b0110, 4.0),
            (0b0111, 6.0),
            (0b1000, -0.0),
            (0b1001, -0.5),
            (0b1011, -1.5),
            (0b1111, -6.0),
        ];
        for (bits, want) in cases {
            assert_eq!(e2m1_to_f32(bits), want, "nibble {bits:04b}");
        }
    }

    #[test]
    fn e4m3_matches_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // e=7, m=0
        assert_eq!(e4m3_to_f32(0x3C), 1.5); // e=7, m=4
        assert_eq!(e4m3_to_f32(0x40), 2.0); // e=8, m=0
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        // largest finite: e=15, m=6 -> 448
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
    }

    #[test]
    fn bf16_widening_is_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }
}
