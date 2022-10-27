// BSL 1.0/Apache 2.0 License

//! A set of generic tuple primitives that may be optimized using SIMD.
//!
//! This crate provides two types: [`Double`] and [`Quad`].

#![cfg_attr(not(breadsimd_no_nightly), allow(incomplete_features))]
#![cfg_attr(not(breadsimd_no_nightly), feature(specialization))]
#![no_std]

cfg_if::cfg_if! {
    // If we don't support SIMD, just use the stable implementation.
    if #[cfg(all(
        not(breadsimd_no_nightly),
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64",
        ),
    ))] {
        mod optimized;
        use optimized as imp;
    } else {
        mod stable;
        use stable as imp;
    }
}

use core::fmt;
use core::ops;

/// A set of two values that may be SIMD optimized.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Double<T: Copy>(imp::Double<T>);

/// A set of four values that may be SIMD optimized.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Quad<T: Copy>(imp::Quad<T>);

macro_rules! implementation {
    (
        $gen:ident,
        $name:ty,
        $self_ident:ident,
        $len:expr,
        [$($index:literal),*]
    ) => {
        impl<$gen: Copy + fmt::Debug> fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }

        impl<$gen: Copy + ops::Add<Output = $gen>> ops::Add for $name {
            type Output = Self;

            fn add(self, other: Self) -> Self::Output {
                $self_ident(self.0 + other.0)
            }
        }

        impl<$gen: Copy + ops::Add<Output = $gen>> ops::AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                self.0 = self.0 + other.0;
            }
        }

        impl<$gen: Copy + ops::Sub<Output = $gen>> ops::Sub for $name {
            type Output = Self;

            fn sub(self, other: Self) -> Self::Output {
                $self_ident(self.0 - other.0)
            }
        }

        impl<$gen: Copy + ops::Sub<Output = $gen>> ops::SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                self.0 = self.0 - other.0;
            }
        }

        impl<$gen: Copy + ops::Mul<Output = $gen>> ops::Mul for $name {
            type Output = Self;

            fn mul(self, other: Self) -> Self::Output {
                $self_ident(self.0 * other.0)
            }
        }

        impl<$gen: Copy + ops::Mul<Output = $gen>> ops::MulAssign for $name {
            fn mul_assign(&mut self, other: Self) {
                self.0 = self.0 * other.0;
            }
        }

        impl<$gen: Copy + ops::Div<Output = $gen>> ops::Div for $name {
            type Output = Self;

            fn div(self, other: Self) -> Self::Output {
                $self_ident(self.0 / other.0)
            }
        }

        impl<$gen: Copy + ops::Div<Output = $gen>> ops::DivAssign for $name {
            fn div_assign(&mut self, other: Self) {
                self.0 = self.0 / other.0;
            }
        }

        impl<$gen: Copy + ops::BitAnd<Output = $gen>> ops::BitAnd for $name {
            type Output = Self;

            fn bitand(self, other: Self) -> Self::Output {
                $self_ident(self.0 & other.0)
            }
        }

        impl<$gen: Copy + ops::BitAnd<Output = $gen>> ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, other: Self) {
                self.0 = self.0 & other.0;
            }
        }

        impl<$gen: Copy + ops::BitOr<Output = $gen>> ops::BitOr for $name {
            type Output = Self;

            fn bitor(self, other: Self) -> Self::Output {
                $self_ident(self.0 | other.0)
            }
        }

        impl<$gen: Copy + ops::BitOr<Output = $gen>> ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, other: Self) {
                self.0 = self.0 | other.0;
            }
        }

        impl<$gen: Copy + ops::BitXor<Output = $gen>> ops::BitXor for $name {
            type Output = Self;

            fn bitxor(self, other: Self) -> Self::Output {
                $self_ident(self.0 ^ other.0)
            }
        }

        impl<$gen: Copy + ops::BitXor<Output = $gen>> ops::BitXorAssign for $name {
            fn bitxor_assign(&mut self, other: Self) {
                self.0 = self.0 ^ other.0;
            }
        }

        impl<$gen: Copy + ops::Not<Output = $gen>> ops::Not for $name {
            type Output = Self;

            fn not(self) -> Self::Output {
                $self_ident(!self.0)
            }
        }

        impl<$gen: Copy> From<[$gen; $len]> for $name {
            fn from(array: [$gen; $len]) -> Self {
                $self_ident(array.into())
            }
        }

        impl<$gen: Copy> ops::Index<usize> for $name {
            type Output = $gen;

            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<$gen: Copy> ops::IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<$gen: Copy> $name {
            /// Create a new array from an array.
            pub fn new(array: [$gen; $len]) -> Self {
                $self_ident(array.into())
            }

            /// Create a new array populated with a single value in all lanes.
            pub fn splat(value: $gen) -> Self {
                $self_ident(imp::$self_ident::splat(value))
            }

            /// Get the underlying array.
            pub fn into_inner(self) -> [$gen; $len] {
                self.0.into_inner()
            }
        }
    };
}

implementation! {
    T,
    Double<T>,
    Double,
    2,
    [0, 1]
}

implementation! {
    T,
    Quad<T>,
    Quad,
    4,
    [0, 1, 2, 3]
}
