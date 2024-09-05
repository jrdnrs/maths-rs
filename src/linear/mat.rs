use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::vec::*;

pub trait Matrix:
    Index<usize, Output = Self::Column>
    + IndexMut<usize, Output = Self::Column>
    + Add<Output = Self>
    + Add<f32, Output = Self>
    + AddAssign
    + AddAssign<f32>
    + Sub<Output = Self>
    + Sub<f32, Output = Self>
    + SubAssign
    + SubAssign<f32>
    + Mul<Output = Self>
    + Mul<f32, Output = Self>
    + MulAssign
    + MulAssign<f32>
    + Div<f32, Output = Self>
    + DivAssign<f32>
    + Neg<Output = Self>
    + Sized
    + Copy
    + Clone
    + Default
    + PartialEq
{
    type Column;
    const ELEMENTS: usize;
    const ZERO: Self;
    const IDENTITY: Self;
    fn uniform(a: f32) -> Self;
    fn transpose(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn abs(&self) -> Self;
    fn recip(&self) -> Self;
    fn round(&self) -> Self;
    fn trunc(&self) -> Self;
    fn fract(&self) -> Self;
    fn floor(&self) -> Self;
    fn ceil(&self) -> Self;
}

macro_rules! assert_near_eq {
    ($x: expr, $y: expr, $delta: expr) => {
        for (xv, yv) in $x.as_array().iter().zip($y.as_array().iter()) {
            assert!((xv - yv).abs() < $delta, "{:?} != {:?}", xv, yv);
        }
    };
}

#[cfg_attr(rustfmt, rustfmt_skip)]
macro_rules! tranpose_mat {
    ($mat: ident, Vec2f) => {
        [
            $mat.array[0], $mat.array[2],
            $mat.array[1], $mat.array[3],
        ]
    };

    ($mat: ident, Vec3f) => {
        [
            $mat.array[0], $mat.array[3], $mat.array[6],
            $mat.array[1], $mat.array[4], $mat.array[7],
            $mat.array[2], $mat.array[5], $mat.array[8],
        ]
    };

    ($mat: ident, Vec4f) => {
        [
            $mat.array[0], $mat.array[4], $mat.array[8], $mat.array[12],
            $mat.array[1], $mat.array[5], $mat.array[9], $mat.array[13],
            $mat.array[2], $mat.array[6], $mat.array[10], $mat.array[14],
            $mat.array[3], $mat.array[7], $mat.array[11], $mat.array[15],
        ]
    };
}

macro_rules! def_mat {
    ( $name: ident, $vec: ident, $dim: expr ) => {
        #[repr(C)]
        #[derive(Debug, Default, Clone, Copy, PartialEq)]
        pub struct $name {
            array: [f32; Self::ELEMENTS],
        }
    };
}

macro_rules! impl_core_mat {
    ( $name: ident, $vec: ident, $dim: expr ) => {
        impl $name {
            pub const ELEMENTS: usize = $dim * $dim;

            pub const ZERO: Self = Self {
                array: [0.0; Self::ELEMENTS],
            };

            pub const IDENTITY: Self = {
                let mut mat = Self::ZERO;
                let mut i = 0;
                while i < $dim {
                    mat.array[i * $dim + i] = 1.0;
                    i += 1;
                }
                mat
            };

            pub fn new(a: [f32; Self::ELEMENTS]) -> Self {
                Self { array: a }
            }

            pub fn from_columns(columns: [$vec; $dim]) -> Self {
                Self {
                    array: unsafe { core::mem::transmute(columns) },
                }
            }

            pub fn from_rows(rows: [$vec; $dim]) -> Self {
                Self::from_columns(rows).transpose()
            }

            pub fn uniform(a: f32) -> Self {
                Self {
                    array: [a; Self::ELEMENTS],
                }
            }

            pub fn transpose(&self) -> Self {
                Self {
                    array: tranpose_mat!(self, $vec),
                }
            }

            pub fn sqrt(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].sqrt();
                }

                matrix
            }

            pub fn abs(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].abs();
                }

                matrix
            }

            pub fn recip(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = if self.array[i] == 0.0 {
                        0.0
                    } else {
                        1.0 / self.array[i]
                    };
                }

                matrix
            }

            pub fn round(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].round();
                }

                matrix
            }

            pub fn trunc(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].trunc();
                }

                matrix
            }

            pub fn fract(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].fract();
                }

                matrix
            }

            pub fn floor(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].floor();
                }

                matrix
            }

            pub fn ceil(&self) -> Self {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i].ceil();
                }

                matrix
            }

            pub fn as_array(&self) -> &[f32; Self::ELEMENTS] {
                &self.array
            }

            pub fn as_mut_array(&mut self) -> &mut [f32; Self::ELEMENTS] {
                &mut self.array
            }

            pub fn as_columns(&self) -> &[$vec; $dim] {
                unsafe { core::mem::transmute(&self.array) }
            }

            pub fn as_mut_columns(&mut self) -> &mut [$vec; $dim] {
                unsafe { core::mem::transmute(&mut self.array) }
            }
        }

        impl Matrix for $name {
            type Column = $vec;

            const ELEMENTS: usize = Self::ELEMENTS;
            const ZERO: Self = Self::ZERO;
            const IDENTITY: Self = Self::IDENTITY;

            fn uniform(a: f32) -> Self {
                Self::uniform(a)
            }

            fn transpose(&self) -> Self {
                self.transpose()
            }

            fn sqrt(&self) -> Self {
                self.sqrt()
            }

            fn abs(&self) -> Self {
                self.abs()
            }

            fn recip(&self) -> Self {
                self.recip()
            }
            
            fn round(&self) -> Self {
                self.round()
            }

            fn trunc(&self) -> Self {
                self.trunc()
            }

            fn fract(&self) -> Self {
                self.fract()
            }

            fn floor(&self) -> Self {
                self.floor()
            }

            fn ceil(&self) -> Self {
                self.ceil()
            }
        }

        impl std::ops::Index<usize> for $name {
            type Output = $vec;

            fn index(&self, col: usize) -> &Self::Output {
                self.as_columns().index(col)
            }
        }

        impl std::ops::IndexMut<usize> for $name {
            fn index_mut(&mut self, col: usize) -> &mut Self::Output {
                self.as_mut_columns().index_mut(col)
            }
        }

        impl std::ops::Add for $name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i] + rhs.array[i];
                }

                matrix
            }
        }

        impl std::ops::AddAssign for $name {
            fn add_assign(&mut self, rhs: Self) {
                for i in 0..(Self::ELEMENTS) {
                    self.array[i] += rhs.array[i];
                }
            }
        }

        impl std::ops::Add<f32> for $name {
            type Output = Self;

            fn add(self, rhs: f32) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i] + rhs;
                }

                matrix
            }
        }

        impl std::ops::AddAssign<f32> for $name {
            fn add_assign(&mut self, rhs: f32) {
                for i in 0..(Self::ELEMENTS) {
                    self.array[i] += rhs;
                }
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i] - rhs.array[i];
                }

                matrix
            }
        }

        impl std::ops::SubAssign for $name {
            fn sub_assign(&mut self, rhs: Self) {
                for i in 0..(Self::ELEMENTS) {
                    self.array[i] -= rhs.array[i];
                }
            }
        }

        impl std::ops::Sub<f32> for $name {
            type Output = Self;

            fn sub(self, rhs: f32) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i] - rhs;
                }

                matrix
            }
        }

        impl std::ops::SubAssign<f32> for $name {
            fn sub_assign(&mut self, rhs: f32) {
                for i in 0..(Self::ELEMENTS) {
                    self.array[i] -= rhs;
                }
            }
        }

        impl std::ops::Mul for $name {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..$dim {
                    for j in 0..$dim {
                        for k in 0..$dim {
                            matrix[j][i] += self[k][i] * rhs[j][k];
                        }
                    }
                }

                matrix
            }
        }

        impl std::ops::MulAssign for $name {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl std::ops::Mul<f32> for $name {
            type Output = Self;

            fn mul(self, rhs: f32) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i] * rhs;
                }

                matrix
            }
        }

        impl std::ops::MulAssign<f32> for $name {
            fn mul_assign(&mut self, rhs: f32) {
                for i in 0..(Self::ELEMENTS) {
                    self.array[i] *= rhs;
                }
            }
        }

        impl std::ops::Mul<$vec> for $name {
            type Output = $vec;

            fn mul(self, rhs: $vec) -> Self::Output {
                let mut vector = $vec::ZERO;

                for i in 0..$dim {
                    for j in 0..$dim {
                        vector[i] += self[j][i] * rhs[j];
                    }
                }

                vector
            }
        }

        impl std::ops::Div<f32> for $name {
            type Output = Self;

            fn div(self, rhs: f32) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = self.array[i] / rhs;
                }

                matrix
            }
        }

        impl std::ops::DivAssign<f32> for $name {
            fn div_assign(&mut self, rhs: f32) {
                for i in 0..(Self::ELEMENTS) {
                    self.array[i] /= rhs;
                }
            }
        }

        impl std::ops::Neg for $name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                let mut matrix = Self::ZERO;

                for i in 0..(Self::ELEMENTS) {
                    matrix.array[i] = -self.array[i];
                }

                matrix
            }
        }

        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Zeroable for $name {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Pod for $name {}
    };
}

def_mat! { Mat2f, Vec2f, 2 }
impl_core_mat! { Mat2f, Vec2f, 2 }

#[cfg_attr(rustfmt, rustfmt_skip)]
impl Mat2f {
    pub fn rotation(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();

        Self {
            array: [
                cos,  sin,
                -sin,  cos,
            ]
        }
    }

    pub fn scale(scale: Vec2f) -> Self {
        Self {
            array: [
                scale.x, 0.0,
                0.0, scale.y,
            ]
        }
    }
    
    pub fn determinant(&self) -> f32 {
        self[0][0] * self[1][1] - self[1][0] * self[0][1]
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();

        if det == 0.0 {
            panic!("Matrix is not invertible");
        }

        let mut matrix = Self::ZERO;

        matrix[0][0] = self[1][1];
        matrix[0][1] = -self[0][1];
        matrix[1][0] = -self[1][0];
        matrix[1][1] = self[0][0];

        let denom = 1.0 / det;
        matrix *= denom;

        return matrix;
    }
}

def_mat! { Mat3f, Vec3f, 3 }
impl_core_mat! { Mat3f, Vec3f, 3 }

#[cfg_attr(rustfmt, rustfmt_skip)]
impl Mat3f {
    pub fn determinant(&self) -> f32 {
        let (a, b, c) = (self[0][0], self[0][1], self[0][2]);
        let (d, e, f) = (self[1][0], self[1][1], self[1][2]);
        let (g, h, i) = (self[2][0], self[2][1], self[2][2]);

          a * (e * i - f * h) 
        - b * (d * i - f * g) 
        + c * (d * h - e * g)
    }

    pub fn inverse(&self) -> Self {
        let (a, b, c) = (self[0][0], self[0][1], self[0][2]);
        let (d, e, f) = (self[1][0], self[1][1], self[1][2]);
        let (g, h, i) = (self[2][0], self[2][1], self[2][2]);

        let det =   a * (e * i - f * h) 
                       - b * (d * i - f * g) 
                       + c * (d * h - e * g);

        if det == 0.0 {
            panic!("Matrix is not invertible");
        }

        let mut matrix = Self::ZERO;

        matrix[0][0] =   e * i - f * h;
        matrix[0][1] = -(b * i - c * h);
        matrix[0][2] =   b * f - c * e;
        matrix[1][0] = -(d * i - f * g);
        matrix[1][1] =   a * i - c * g;
        matrix[1][2] = -(a * f - c * d);
        matrix[2][0] =   d * h - e * g;
        matrix[2][1] = -(a * h - b * g);
        matrix[2][2] =   a * e - b * d;

        let denom = 1.0 / det;
        matrix *= denom;

        return matrix;
    }
}

def_mat! { Mat4f, Vec4f, 4 }
impl_core_mat! { Mat4f, Vec4f, 4 }

#[cfg_attr(rustfmt, rustfmt_skip)]
impl Mat4f {
    pub fn determinant(&self) -> f32 {
        let (a, b, c, d) = (self[0][0], self[0][1], self[0][2], self[0][3]);
        let (e, f, g, h) = (self[1][0], self[1][1], self[1][2], self[1][3]);
        let (i, j, k, l) = (self[2][0], self[2][1], self[2][2], self[2][3]);
        let (m, n, o, p) = (self[3][0], self[3][1], self[3][2], self[3][3]);

        let kp_ol = k * p - o * l;
        let jp_nl = j * p - n * l;
        let jo_nk = j * o - n * k;
        let ip_ml = i * p - m * l;
        let io_mk = i * o - m * k;
        let in_mj = i * n - m * j;

          a * (f * kp_ol - g * jp_nl + h * jo_nk)
        - b * (e * kp_ol - g * ip_ml + h * io_mk)
        + c * (e * jp_nl - f * ip_ml + h * in_mj)
        - d * (e * jo_nk - f * io_mk + g * in_mj)
    }

    pub fn inverse(&self) -> Self {
        let (a, b, c, d) = (self[0][0], self[0][1], self[0][2], self[0][3]);
        let (e, f, g, h) = (self[1][0], self[1][1], self[1][2], self[1][3]);
        let (i, j, k, l) = (self[2][0], self[2][1], self[2][2], self[2][3]);
        let (m, n, o, p) = (self[3][0], self[3][1], self[3][2], self[3][3]);

        let kp_ol = k * p - o * l;
        let jp_nl = j * p - n * l;
        let jo_nk = j * o - n * k;
        let ip_ml = i * p - m * l;
        let io_mk = i * o - m * k;
        let in_mj = i * n - m * j;
        
        let m00 = f * kp_ol - g * jp_nl + h * jo_nk;
        let m10 = e * kp_ol - g * ip_ml + h * io_mk;
        let m20 = e * jp_nl - f * ip_ml + h * in_mj;
        let m30 = e * jo_nk - f * io_mk + g * in_mj;

        let det =   a * m00 
                       - b * m10
                       + c * m20
                       - d * m30;

        if det == 0.0 {
            panic!("Matrix is not invertible");
        }

        let mut matrix = Self::ZERO;

        let gp_oh = g * p - o * h;
        let fp_nh = f * p - n * h;
        let fo_ng = f * o - n * g;
        let ep_mh = e * p - m * h;
        let eo_mg = e * o - m * g;
        let en_mf = e * n - m * f;

        let gl_kh = g * l - k * h;
        let fl_jh = f * l - j * h;
        let fk_jg = f * k - j * g;
        let el_ih = e * l - i * h;
        let ek_ig = e * k - i * g;
        let ej_if = e * j - i * f;

        matrix[0][0] =   m00;
        matrix[0][1] = -(b * kp_ol - c * jp_nl + d * jo_nk);
        matrix[0][2] =   b * gp_oh - c * fp_nh + d * fo_ng;
        matrix[0][3] = -(b * gl_kh - c * fl_jh + d * fk_jg);
        matrix[1][0] = - m10;
        matrix[1][1] =   a * kp_ol - c * ip_ml + d * io_mk;
        matrix[1][2] = -(a * gp_oh - c * ep_mh + d * eo_mg);
        matrix[1][3] =   a * gl_kh - c * el_ih + d * ek_ig;
        matrix[2][0] =   m20;
        matrix[2][1] = -(a * jp_nl - b * ip_ml + d * in_mj);
        matrix[2][2] =   a * fp_nh - b * ep_mh + d * en_mf;
        matrix[2][3] = -(a * fl_jh - b * el_ih + d * ej_if);
        matrix[3][0] = - m30;
        matrix[3][1] =   a * jo_nk - b * io_mk + c * in_mj;
        matrix[3][2] = -(a * fo_ng - b * eo_mg + c * en_mf);
        matrix[3][3] =   a * fk_jg - b * ek_ig + c * ej_if;


        let denom = 1.0 / det;
        matrix *= denom;

        matrix
    }

    /// Create a (left-handed) perspective projection matrix for DirectX, where:
    /// - X is right
    /// - Y is up
    /// - Z is forward
    /// 
    /// NDC: [-1, 1] for x and y, [0, 1] for z
    pub fn perspective_directx(aspect_ratio: f32, fov_rad: f32, near: f32, far: f32) -> Self {
        let a = aspect_ratio;
        let f = 1.0 / (fov_rad / 2.0).tan();
        let inv_fn = 1.0 / (far - near);

        // transformed Z, Zt = (g * Z + h) / Z
        // division by Z occurs in graphics pipeline
        let g =           far * inv_fn;
        let h = -(far * near) * inv_fn;

        let mut matrix = Mat4f::ZERO;
        
        matrix[0][0] = f / a;
        matrix[1][1] = -f;
        matrix[2][2] = g;
        matrix[3][2] = h;
        matrix[2][3] = 1.0;

        matrix
    }

    /// Creates a (left-handed) perspective projection matrix for OpenGL, where:
    /// - X is right
    /// - Y is up
    /// - Z is forward
    /// 
    /// NDC: [-1, 1]
    pub fn perspective_opengl(aspect_ratio: f32, fov_rad: f32, near: f32, far: f32) -> Self {
        let a = aspect_ratio;
        let f = 1.0 / (fov_rad / 2.0).tan();
        let inv_fn = 1.0 / (far - near);

        // transformed Z, Zt = (g * Z + h) / Z
        // division by Z occurs in graphics pipeline
        let g =      -(far + near) * inv_fn;
        let h = (2.0 * far * near) * inv_fn;

        let mut matrix = Mat4f::ZERO;
        
        matrix[0][0] = f / a;
        matrix[1][1] = -f;
        matrix[2][2] = g;
        matrix[3][2] = h;
        matrix[2][3] = 1.0;

        matrix
    }

    pub fn orthographic_opengl(size: f32, near: f32, far: f32) -> Self {
        // let a = aspect_ratio;
        let w = size;
        let h = w ;

        let r =  w * 0.5;
        let l = -r;
        let t =  h * 0.5;
        let b = -t;
        
        let inv_rl = 1.0 / (r - l);
        let inv_tb = 1.0 / (t - b);
        let inv_fn = 1.0 / (far - near);

        let mut matrix = Mat4f::IDENTITY;
        
        matrix[0][0] =           2.0 * inv_rl;
        matrix[3][0] =      -(r + l) * inv_rl;
        matrix[1][1] =           2.0 * inv_tb;
        matrix[3][1] =      -(t + b) * inv_tb;
        matrix[2][2] =          -2.0 * inv_fn;
        matrix[3][2] = -(far + near) * inv_fn;

        matrix
    }

    pub fn translate(x: f32, y: f32, z: f32) -> Self {
        let mut matrix = Mat4f::IDENTITY;
        
        matrix[3][0] = x;
        matrix[3][1] = y;
        matrix[3][2] = z;

        matrix
    }

    pub fn scale(x: f32, y: f32, z: f32) -> Self {
        let mut matrix = Mat4f::IDENTITY;
        
        matrix[0][0] = x;
        matrix[1][1] = y;
        matrix[2][2] = z;

        matrix
    }

    pub fn rotate(rad: f32, direction: &Vec3f) -> Self {
        let d = direction.normalise();

        let cos = rad.cos();
        let omcos= 1.0 - cos;
        let sin = rad.sin();


        let mut matrix = Mat4f::IDENTITY;
        
        matrix[0][0] = cos + d.x*d.x * omcos;
        matrix[1][0] = d.x * d.y     * omcos - (d.z * sin);
        matrix[2][0] = d.x * d.z     * omcos + (d.y * sin);
        matrix[0][1] = d.y * d.x     * omcos + (d.z * sin);
        matrix[1][1] = cos + d.y*d.y * omcos;
        matrix[2][1] = d.y * d.z     * omcos - (d.x * sin);
        matrix[0][2] = d.z * d.x     * omcos - (d.y * sin);
        matrix[1][2] = d.z * d.y     * omcos + (d.x * sin);
        matrix[2][2] = cos + d.z*d.z * omcos;

        matrix
    }

    pub fn rotate_around_x(rad: f32) -> Self {
        let cos = rad.cos();
        let sin = rad.sin();

        let mut matrix = Mat4f::IDENTITY;
        
        matrix[1][1] =  cos;
        matrix[2][1] =  sin;
        matrix[1][2] = -sin;
        matrix[2][2] =  cos;

        matrix
    }

    pub fn rotate_around_y(rad: f32) -> Self {
        let cos = rad.cos();
        let sin = rad.sin();

        let mut matrix = Mat4f::IDENTITY;
        
        matrix[0][0] =  cos;
        matrix[2][0] = -sin;
        matrix[0][2] =  sin;
        matrix[2][2] =  cos;

        matrix
    }

    pub fn rotate_around_z(rad: f32) -> Self {
        let cos = rad.cos();
        let sin = rad.sin();

        let mut matrix = Mat4f::IDENTITY;
        
        matrix[0][0] =  cos;
        matrix[1][0] =  sin;
        matrix[0][1] = -sin;
        matrix[1][1] =  cos;

        matrix
    }

    pub fn transform_point(&self, point: &Vec3f) -> Vec3f {
        let vector = Vec4f::new(point.x, point.y, point.z, 1.0);
        let transformed = *self * vector;
        return Vec3f::new(transformed.x, transformed.y, transformed.z);
    }
}

impl From<Mat3f> for Mat4f {
    fn from(mat: Mat3f) -> Self {
        let mut matrix = Self::IDENTITY;

        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] = mat[i][j];
            }
        }

        return matrix;
    }
}

impl From<Mat4f> for Mat3f {
    fn from(mat: Mat4f) -> Self {
        let mut matrix = Self::IDENTITY;

        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] = mat[i][j];
            }
        }

        return matrix;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat3_from_rows() {
        let mat = Mat3f::from_rows([
            Vec3f::new(1.0, 2.0, 3.0),
            Vec3f::new(4.0, 5.0, 6.0),
            Vec3f::new(7.0, 8.0, 9.0),
        ]);
        assert_eq!(mat[0], Vec3f::new(1.0, 4.0, 7.0));
        assert_eq!(mat[1], Vec3f::new(2.0, 5.0, 8.0));
        assert_eq!(mat[2], Vec3f::new(3.0, 6.0, 9.0));
    }

    #[test]
    fn mat3_identity() {
        let mat = Mat3f::IDENTITY;
        assert_eq!(mat[0], Vec3f::new(1.0, 0.0, 0.0));
        assert_eq!(mat[1], Vec3f::new(0.0, 1.0, 0.0));
        assert_eq!(mat[2], Vec3f::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn mat3_mul() {
        let m1 = Mat3f::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let v1 = Vec3f::new(1.0, 2.0, 3.0);
        assert_eq!(m1 * v1, Vec3f::new(30.0, 36.0, 42.0));

        let m2 = Mat3f::new([13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 11.0, 12.0]);
        assert_eq!(
            m1 * m2,
            Mat3f::new([174.0, 216.0, 258.0, 210.0, 261.0, 312.0, 147.0, 189.0, 231.0])
        );
    }

    #[test]
    fn mat3_transpose() {
        let mat = Mat3f::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(
            mat.transpose(),
            Mat3f::new([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])
        );
    }

    #[test]
    fn mat3_determinant() {
        let mat = Mat3f::new([1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0]);
        assert_eq!(mat.determinant(), 22.0);
    }

    #[test]
    fn mat3_inverse() {
        let a = Mat3f::new([1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0]);
        assert_eq!(
            a.inverse(),
            Mat3f::new([
                12.0 / 11.0,
                -6.0 / 11.0,
                -1.0 / 11.0,
                5.0 / 22.0,
                3.0 / 22.0,
                -5.0 / 22.0,
                -2.0 / 11.0,
                1.0 / 11.0,
                2.0 / 11.0
            ])
        );

        let b = Mat3f::new([5.0, 7.0, 1.0, 18.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            b.inverse(),
            Mat3f::new([
                3.0 / 605.0,
                37.0 / 605.0,
                -19.0 / 605.0,
                96.0 / 605.0,
                -26.0 / 605.0,
                -3.0 / 605.0,
                -82.0 / 605.0,
                -3.0 / 605.0,
                116.0 / 605.0
            ])
        );
    }

    #[test]
    #[should_panic]
    fn mat3_inverse_panic() {
        let a = Mat3f::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        a.inverse();
    }

    #[test]
    fn mat4_identity() {
        let mat = Mat4f::IDENTITY;
        assert_eq!(mat[0], Vec4f::new(1.0, 0.0, 0.0, 0.0));
        assert_eq!(mat[1], Vec4f::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(mat[2], Vec4f::new(0.0, 0.0, 1.0, 0.0));
        assert_eq!(mat[3], Vec4f::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn mat4_mul() {
        let m1 = Mat4f::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let v1 = Vec4f::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m1 * v1, Vec4f::new(90.0, 100.0, 110.0, 120.0));

        let m2 = Mat4f::new([
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0, 19.0,
        ]);
        assert_eq!(
            m1 * m2,
            Mat4f::new([
                426.0, 484.0, 542.0, 600.0, 421.0, 486.0, 551.0, 616.0, 398.0, 452.0, 506.0, 560.0,
                510.0, 580.0, 650.0, 720.0
            ])
        );
    }

    #[test]
    fn mat4_transpose() {
        let mat = Mat4f::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10., 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        assert_eq!(
            mat.transpose(),
            Mat4f::new([
                1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10., 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
                16.0
            ])
        );
    }

    #[test]
    fn mat4_determinant() {
        let mat = Mat4f::new([
            1.0, 2.0, 3.0, 4.0, 0.0, 4.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 1.0, 2.0, 0.0, 8.0,
        ]);
        assert_eq!(mat.determinant(), 160.0);
    }

    #[test]
    fn mat4_inverse() {
        let a = Mat4f::new([
            1.0, 2.0, 3.0, 4.0, 0.0, 4.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 1.0, 2.0, 0.0, 8.0,
        ]);
        assert_near_eq!(
            a.inverse(),
            Mat4f::new([
                19.0 / 16.0,
                -9.0 / 16.0,
                -1.0 / 8.0,
                -1.0 / 16.0,
                41.0 / 160.0,
                21.0 / 160.0,
                -19.0 / 80.0,
                -3.0 / 160.0,
                1.0 / 20.0,
                1.0 / 20.0,
                1.0 / 10.0,
                -3.0 / 20.0,
                -17.0 / 80.0,
                3.0 / 80.0,
                3.0 / 40.0,
                11.0 / 80.0
            ]),
            0.000001
        );

        let b = Mat4f::new([
            18.0, 7.0, 1.0, 18.0, 2.0, 3.0, 4.0, 1.0, 6.0, 7.0, 13.0, 2.0, 3.0, 7.0, 5.0, 9.0,
        ]);
        assert_near_eq!(
            b.inverse(),
            Mat4f::new([
                69.0 / 1144.0,
                7.0 / 26.0,
                -37.0 / 1144.0,
                -41.0 / 286.0,
                -1.0 / 88.0,
                3.0 / 2.0,
                -39.0 / 88.0,
                -1.0 / 22.0,
                -25.0 / 1144.0,
                -21.0 / 26.0,
                345.0 / 1144.0,
                19.0 / 286.0,
                1.0 / 1144.0,
                -21.0 / 26.0,
                215.0 / 1144.0,
                45.0 / 286.0
            ]),
            0.000001
        );
    }
}
