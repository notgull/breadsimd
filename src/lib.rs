// BSL 1.0/Apache 2.0 License

//! A set of generic tuple primitives that may be optimized using SIMD.
//!
//! This crate provides two types: [`Double`] and [`Quad`]. For all intents and purposes,
//! [`Double`] is equivalent to a `[T; 2]` and [`Quad`] is equivalent to a `[T; 4]`.
//! In fact, on Stable Rust, they are just thin wrappers around arrays.
//! 
//! However, if this crate is compiled with Nightly Rust, in certain cases they will b
//! replaced with SIMD types. Using specialization, on certain platforms these types
//! will be optimized using SIMD:
//! 
//! - `i32`
//! - `u32`
//! - `f32`
//! 
//! This is not a guarantee. More types may be added or removed in the future, which will
//! be signified with a bump in the patch version.
//! 
//! These architecture will be optimized if SIMD is detected during compile time:
//! 
//! - `x86`
//! - `x86_64`
//! - `arm`
//! - `aarch64`
//! 
//! As with the above, this is not a guarantee. More architectures may be added or removed
//! in the future, which will be signified with a bump in the patch version.
//! 
//! ## Goals
//! 
//! The goal of this crate is to let users have their cake and eat it, too. You can write
//! code using [`Double`] and [`Quad`] without worrying about whether or not they are 
//! optimized using SIMD. If they can be optimized properly, they will. If not, it will
//! fall back to the generic implementation.
//! 
//! The primary use case for this crate is in geometry libraries. [`Double`] is intended
//! to represent a single point, while [`Rect`] is intended to represent a rectangle.
//! However, it's likely that this crate will be useful in other areas as well.
//! 
//! This crate is also `no_std`, allowing it to be used seamlessly on embedded platforms.
//! 
//! ## Example
//! 
//! ```
//! use breadsimd::Double;
//! use core::num::Wrapping;
//! 
//! // At the time of writing, `Wrapping<u32>` is not optimized using SIMD.
//! let mut a = Double::new([Wrapping(1), Wrapping(2)]);
//! a += Double::new([Wrapping(3), Wrapping(4)]);
//! assert_eq!(a, Double::new([Wrapping(4), Wrapping(6)]));
//! 
//! // However, `u32` is optimized using SIMD.
//! let mut b = Double::<u32>::new([1, 2]);
//! b += Double::new([3, 4]);
//! assert_eq!(b, Double::new([4, 6])); 
//! ```

#![cfg_attr(not(breadsimd_no_nightly), allow(incomplete_features))]
#![cfg_attr(not(breadsimd_no_nightly), feature(specialization))]
#![no_std]

#[cfg(test)]
extern crate std;

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
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }

        impl<$gen: Copy + ops::Add<Output = $gen>> ops::Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, other: Self) -> Self::Output {
                $self_ident(self.0 + other.0)
            }
        }

        impl<$gen: Copy + ops::Add<Output = $gen>> ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.0 = self.0 + other.0;
            }
        }

        impl<$gen: Copy + ops::Sub<Output = $gen>> ops::Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, other: Self) -> Self::Output {
                $self_ident(self.0 - other.0)
            }
        }

        impl<$gen: Copy + ops::Sub<Output = $gen>> ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.0 = self.0 - other.0;
            }
        }

        impl<$gen: Copy + ops::Mul<Output = $gen>> ops::Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, other: Self) -> Self::Output {
                $self_ident(self.0 * other.0)
            }
        }

        impl<$gen: Copy + ops::Mul<Output = $gen>> ops::MulAssign for $name {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                self.0 = self.0 * other.0;
            }
        }

        impl<$gen: Copy + ops::Div<Output = $gen>> ops::Div for $name {
            type Output = Self;

            #[inline]
            fn div(self, other: Self) -> Self::Output {
                $self_ident(self.0 / other.0)
            }
        }

        impl<$gen: Copy + ops::Div<Output = $gen>> ops::DivAssign for $name {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                self.0 = self.0 / other.0;
            }
        }

        impl<$gen: Copy + ops::BitAnd<Output = $gen>> ops::BitAnd for $name {
            type Output = Self;

            #[inline]
            fn bitand(self, other: Self) -> Self::Output {
                $self_ident(self.0 & other.0)
            }
        }

        impl<$gen: Copy + ops::BitAnd<Output = $gen>> ops::BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, other: Self) {
                self.0 = self.0 & other.0;
            }
        }

        impl<$gen: Copy + ops::BitOr<Output = $gen>> ops::BitOr for $name {
            type Output = Self;

            #[inline]
            fn bitor(self, other: Self) -> Self::Output {
                $self_ident(self.0 | other.0)
            }
        }

        impl<$gen: Copy + ops::BitOr<Output = $gen>> ops::BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, other: Self) {
                self.0 = self.0 | other.0;
            }
        }

        impl<$gen: Copy + ops::BitXor<Output = $gen>> ops::BitXor for $name {
            type Output = Self;

            #[inline]
            fn bitxor(self, other: Self) -> Self::Output {
                $self_ident(self.0 ^ other.0)
            }
        }

        impl<$gen: Copy + ops::BitXor<Output = $gen>> ops::BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, other: Self) {
                self.0 = self.0 ^ other.0;
            }
        }

        impl<$gen: Copy + ops::Not<Output = $gen>> ops::Not for $name {
            type Output = Self;

            #[inline]
            fn not(self) -> Self::Output {
                $self_ident(!self.0)
            }
        }

        impl<$gen: Copy> From<[$gen; $len]> for $name {
            #[inline]
            fn from(array: [$gen; $len]) -> Self {
                $self_ident(array.into())
            }
        }

        impl<$gen: Copy> ops::Index<usize> for $name {
            type Output = $gen;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<$gen: Copy> ops::IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<$gen: Copy> $name {
            /// Create a new array from an array.
            #[inline]
            pub fn new(array: [$gen; $len]) -> Self {
                $self_ident(array.into())
            }

            /// Create a new array populated with a single value in all lanes.
            #[inline]
            pub fn splat(value: $gen) -> Self {
                $self_ident(imp::$self_ident::splat(value))
            }

            /// Get the underlying array.
            #[inline]
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
