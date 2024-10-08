use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub trait Vector:
    Index<usize, Output = f32>
    + IndexMut<usize, Output = f32>
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
    + Div<Output = Self>
    + Div<f32, Output = Self>
    + DivAssign
    + DivAssign<f32>
    + Neg<Output = Self>
    + Sized
    + Copy
    + Clone
    + Default
    + PartialEq
{
    const ELEMENTS: usize;
    const ZERO: Self;
    fn uniform(a: f32) -> Self;
    fn normalise(&self) -> Self;
    fn magnitude(&self) -> f32;
    fn magnitude_sq(&self) -> f32;
    fn dot(&self, rhs: Self) -> f32;
    fn sqrt(&self) -> Self;
    fn abs(&self) -> Self;
    fn recip(&self) -> Self;
    fn round(&self) -> Self;
    fn trunc(&self) -> Self;
    fn fract(&self) -> Self;
    fn floor(&self) -> Self;
    fn ceil(&self) -> Self;
    fn min(&self, rhs: &Self) -> Self;
    fn max(&self, rhs: &Self) -> Self;
    fn lerp(&self, rhs: Self, t: f32) -> Self;
}

macro_rules! count_items {
    ($name:ident) => { 1 };
    ($first:ident, $($rest:ident),*) => {
        1 + count_items!($($rest),*)
    }
}

/// We can't use '+' as the separator, so we include it at the start of the
/// repetition and then strip the initial '+'.
macro_rules! strip_plus {
    {+ $($rest:tt)* } => { $($rest)* }
}

macro_rules! def_vec {
    ( $name: ident { $($field: ident),+ } ) => {

        #[repr(C)]
        #[derive(Debug, Default, Clone, Copy, PartialEq)]
        pub struct $name {
            $(pub $field: f32),+
        }

    }
}

macro_rules! impl_core_vec {
    ( $name: ident { $($field: ident),+ } ) => {

        impl $name {
            pub const ELEMENTS: usize = count_items!( $($field),+ );

            pub const ZERO: Self = Self {
                $($field: 0.0),+
            };

            pub const ONE: Self = Self {
                $($field: 1.0),+
            };


            pub fn new( $($field: f32),+ ) -> Self {
                Self {
                    $($field),+
                }

            }

            pub fn uniform(a: f32) -> Self {
                Self {
                    $($field: a),+
                }
            }

            pub fn normalise(&self) -> Self {
                let m = self.magnitude();

                if m == 0.0 {
                    return Self::ZERO;
                }

                Self {
                    $($field: self.$field / m),+
                }
            }

            pub fn magnitude(&self) -> f32 {
                (
                    strip_plus!(
                        $(+ self.$field * self.$field)+
                    )
                ).sqrt()
            }

            pub fn magnitude_sq(&self) -> f32 {
                (
                    strip_plus!(
                        $(+ self.$field * self.$field)+
                    )
                )
            }

            pub fn dot(&self, rhs: Self) -> f32 {
                strip_plus!(
                    $(+ self.$field * rhs.$field)+
                )
            }

            pub fn sqrt(&self) -> Self {
                Self {
                    $($field: self.$field.sqrt()),+
                }
            }

            pub fn abs(&self) -> Self {
                Self {
                    $($field: self.$field.abs()),+
                }
            }

            pub fn recip(&self) -> Self {
                Self {
                    $($field: self.$field.recip()),+
                }
            }

            pub fn round(&self) -> Self {
                Self {
                    $($field: self.$field.round()),+
                }
            }

            pub fn trunc(&self) -> Self {
                Self {
                    $($field: self.$field.trunc()),+
                }
            }

            pub fn fract(&self) -> Self {
                Self {
                    $($field: self.$field.fract()),+
                }
            }

            pub fn floor(&self) -> Self {
                Self {
                    $($field: self.$field.floor()),+
                }
            }

            pub fn ceil(&self) -> Self {
                Self {
                    $($field: self.$field.ceil()),+
                }
            }

            pub fn min(&self, rhs: &Self) -> Self {
                Self {
                    $($field: self.$field.min(rhs.$field)),+
                }
            }

            pub fn max(&self, rhs: &Self) -> Self {
                Self {
                    $($field: self.$field.max(rhs.$field)),+
                }
            }

            pub fn lerp(&self, rhs: Self, t: f32) -> Self {
                Self {
                    $($field: self.$field + (rhs.$field - self.$field) * t),+
                }
            }

            pub fn as_array(&self) -> &[f32; Self::ELEMENTS] {
                unsafe {
                    core::mem::transmute::<_, &[f32; Self::ELEMENTS]>(self)
                }
            }

            pub fn as_mut_array(&mut self) -> &mut [f32; Self::ELEMENTS] {
                unsafe {
                    core::mem::transmute::<_, &mut [f32; Self::ELEMENTS]>(self)
                }
            }

        }

        impl Vector for $name {
            const ELEMENTS: usize = Self::ELEMENTS;
            const ZERO: Self = Self::ZERO;

            fn uniform(a: f32) -> Self {
                Self::uniform(a)
            }

            fn normalise(&self) -> Self {
                self.normalise()
            }

            fn magnitude(&self) -> f32 {
                self.magnitude()
            }

            fn magnitude_sq(&self) -> f32 {
                self.magnitude_sq()
            }

            fn dot(&self, rhs: Self) -> f32 {
                self.dot(rhs)
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

            fn min(&self, rhs: &Self) -> Self {
                self.min(rhs)
            }

            fn max(&self, rhs: &Self) -> Self {
                self.max(rhs)
            }

            fn lerp(&self, rhs: Self, t: f32) -> Self {
                self.lerp(rhs, t)
            }
        }


        impl std::ops::Index<usize> for $name {
            type Output = f32;

            fn index(&self, i: usize) -> &Self::Output {
                self.as_array().index(i)
            }
        }

        impl std::ops::IndexMut<usize> for $name {
            fn index_mut(&mut self, i: usize) -> &mut Self::Output {
                self.as_mut_array().index_mut(i)
            }
        }

        impl std::ops::Add for $name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self {
                    $($field: self.$field + rhs.$field),+
                }
            }
        }

        impl std::ops::AddAssign for $name {
            fn add_assign(&mut self, rhs: Self) {
                $(self.$field += rhs.$field);+
            }
        }

        impl std::ops::Add<f32> for $name {
            type Output = Self;

            fn add(self, rhs: f32) -> Self::Output {
                Self {
                    $($field: self.$field + rhs),+
                }
            }
        }

        impl std::ops::AddAssign<f32> for $name {
            fn add_assign(&mut self, rhs: f32) {
                $(self.$field += rhs);+
            }
        }

        impl std::ops::Sub for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self {
                    $($field: self.$field - rhs.$field),+
                }
            }
        }

        impl std::ops::SubAssign for $name {
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$field -= rhs.$field);+
            }
        }

        impl std::ops::Sub<f32> for $name {
            type Output = Self;

            fn sub(self, rhs: f32) -> Self::Output {
                Self {
                    $($field: self.$field - rhs),+
                }
            }
        }

        impl std::ops::SubAssign<f32> for $name {
            fn sub_assign(&mut self, rhs: f32) {
                $(self.$field -= rhs);+
            }
        }

        impl std::ops::Mul for $name {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self {
                    $($field: self.$field * rhs.$field),+
                }
            }
        }

        impl std::ops::MulAssign for $name {
            fn mul_assign(&mut self, rhs: Self) {
                $(self.$field *= rhs.$field);+
            }
        }

        impl std::ops::Mul<f32> for $name {
            type Output = Self;

            fn mul(self, rhs: f32) -> Self::Output {
                Self {
                    $($field: self.$field * rhs),+
                }
            }
        }

        impl std::ops::MulAssign<f32> for $name {
            fn mul_assign(&mut self, rhs: f32) {
                $(self.$field *= rhs);+
            }
        }

        impl std::ops::Div for $name {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                Self {
                    $($field: self.$field / rhs.$field),+
                }
            }
        }

        impl std::ops::DivAssign for $name {
            fn div_assign(&mut self, rhs: Self) {
                $(self.$field /= rhs.$field);+
            }
        }

        impl std::ops::Div<f32> for $name {
            type Output = Self;

            fn div(self, rhs: f32) -> Self::Output {
                Self {
                    $($field: self.$field / rhs),+
                }
            }
        }

        impl std::ops::DivAssign<f32> for $name {
            fn div_assign(&mut self, rhs: f32) {
                $(self.$field /= rhs);+
            }
        }

        impl std::ops::Neg for $name {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self {
                    $($field: -self.$field),+
                }
            }
        }

        impl From<[f32; Self::ELEMENTS]> for $name {
            fn from(array: [f32; Self::ELEMENTS]) -> Self {
                unsafe { core::mem::transmute(array) }
            }
        }


        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Zeroable for $name {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Pod for $name {}
    };
}

def_vec! { Vec2f {x, y} }
impl_core_vec! { Vec2f {x, y} }

impl Vec2f {
    pub fn perpendicular(&self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    pub fn reflect(&self, normal: Self) -> Self {
        *self - normal * 2.0 * self.dot(normal)
    }

    pub fn cross(&self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }

    pub fn rotate(&self, sin: f32, cos: f32) -> Self {
        Self {
            x: self.x * cos - self.y * sin,
            y: self.x * sin + self.y * cos,
        }
    }
}

def_vec! { Vec3f {x, y, z} }
impl_core_vec! { Vec3f {x, y, z} }

impl Vec3f {
    pub fn cross(&self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl From<Vec2f> for Vec3f {
    fn from(v: Vec2f) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: 0.0,
        }
    }
}

impl From<Vec3f> for Vec2f {
    fn from(v: Vec3f) -> Self {
        Self { x: v.x, y: v.y }
    }
}

def_vec! { Vec4f {x, y, z, w} }
impl_core_vec! { Vec4f {x, y, z, w} }

impl From<Vec3f> for Vec4f {
    fn from(v: Vec3f) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
            w: 1.0,
        }
    }
}

impl From<Vec4f> for Vec3f {
    fn from(v: Vec4f) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}
