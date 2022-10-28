// Copyright John Nunley, 2022.
//
// This software is distributed under the Boost Software License Version 1.0 and the Apache
// 2.0 License, at your option. See the `LICENSE-BOOST` and `LICENSE-APACHE` files in the
// root of this repository for the full text of the licenses.
//
// --------------------------------------------------------------------------------------------
//
//  Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE-BOOST or copy at
//        https://www.boost.org/LICENSE_1_0.txt)
//
// --------------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A set of generic tuple primitives that may be optimized using SIMD.
//!
//! This crate provides two types: [`Double`] and [`Quad`]. For all intents and purposes,
//! [`Double`] is equivalent to a `[T; 2]` and [`Quad`] is equivalent to a `[T; 4]`.
//! In fact, on Stable Rust, they are just thin wrappers around arrays.
//!
//! However, if this crate is compiled with Nightly Rust, in certain cases they will b
//! replaced with SIMD types. Using specialization, on certain platforms these types
//! will be optimized using SIMD. SIMD is done using the currently-unstable `portable_simd`
//! feature in the standard library. Currently, all of the elementary number types
//! except for `u128` and `i128` are optimized in this way. This optimization includes
//! `f32` and `f64`.
//!
//! ## Goals
//!
//! The goal of this crate is to let users have their cake and eat it, too. You can write
//! code using [`Double`] and [`Quad`] without worrying about whether or not they are
//! optimized using SIMD. If they can be optimized properly, they will. If not, it will
//! fall back to the generic implementation.
//!
//! The primary use case for this crate is in geometry libraries. [`Double`] is intended
//! to represent a single point, while [`Quad`] is intended to represent a rectangle.
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
#![cfg_attr(not(breadsimd_no_nightly), feature(portable_simd, specialization))]
#![forbid(
    unsafe_code,
    future_incompatible,
    missing_docs,
    missing_debug_implementations
)]
#![no_std]
#![warn(
    clippy::pedantic,
    clippy::style,
    clippy::complexity,
    clippy::cargo,
    clippy::perf,
    clippy::correctness
)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(any(test, feature = "std"))]
extern crate std;

cfg_if::cfg_if! {
    // If we don't support SIMD, just use the stable implementation.
    if #[cfg(not(breadsimd_no_nightly))] {
        mod optimized;
        use optimized as imp;
    } else {
        mod stable;
        use stable as imp;
    }
}

use core::fmt;
use core::iter::{Product, Sum};
use core::ops;

/// A set of two values that may be SIMD optimized.
///
/// See the [crate-level documentation](crate) for more information.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Double<T: Copy>(imp::Double<T>);

/// A set of four values that may be SIMD optimized.
///
/// See the [crate-level documentation](crate) for more information.
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

        impl<$gen: Copy + ops::Neg<Output = $gen>> ops::Neg for $name {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                $self_ident(-self.0)
            }
        }

        impl<$gen: Copy + ops::Shl<Output = $gen>> ops::Shl for $name {
            type Output = Self;

            #[inline]
            fn shl(self, other: Self) -> Self::Output {
                $self_ident(self.0 << other.0)
            }
        }

        impl<$gen: Copy + ops::Shl<Output = $gen>> ops::ShlAssign for $name {
            #[inline]
            fn shl_assign(&mut self, other: Self) {
                self.0 = self.0 << other.0;
            }
        }

        impl<$gen: Copy + ops::Shr<Output = $gen>> ops::Shr for $name {
            type Output = Self;

            #[inline]
            fn shr(self, other: Self) -> Self::Output {
                $self_ident(self.0 >> other.0)
            }
        }

        impl<$gen: Copy + ops::Shr<Output = $gen>> ops::ShrAssign for $name {
            #[inline]
            fn shr_assign(&mut self, other: Self) {
                self.0 = self.0 >> other.0;
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

        impl<$gen: Copy> AsRef<[$gen; $len]> for $name {
            #[inline]
            fn as_ref(&self) -> &[$gen; $len] {
                self.0.as_ref()
            }
        }

        impl<$gen: Copy> AsMut<[$gen; $len]> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut [$gen; $len] {
                self.0.as_mut()
            }
        }

        impl<$gen: Copy> AsRef<[$gen]> for $name {
            #[inline]
            fn as_ref(&self) -> &[$gen] {
                self.0.as_ref()
            }
        }

        impl<$gen: Copy> AsMut<[$gen]> for $name {
            #[inline]
            fn as_mut(&mut self) -> &mut [$gen] {
                self.0.as_mut()
            }
        }

        impl<$gen: num_traits::Zero + Copy + ops::Add<Output = $gen>> Sum for $name {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($self_ident::splat($gen::zero()), ops::Add::add)
            }
        }

        impl<$gen: num_traits::One + Copy + ops::Mul<Output = $gen>> Product for $name {
            #[inline]
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($self_ident::splat($gen::one()), ops::Mul::mul)
            }
        }

        impl<$gen: Copy> $name {
            /// Create a new array from an array.
            #[inline]
            pub fn new(array: [$gen; $len]) -> Self {
                $self_ident(imp::$self_ident::new(array))
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

        impl<$gen: num_traits::float::FloatCore> $name {
            /// Get the absolute value of each lane.
            #[must_use]
            #[inline]
            pub fn abs(self) -> Self {
                $self_ident(self.0.abs())
            }

            /// Get the maximum of each lane.
            #[must_use]
            #[inline]
            pub fn max(self, other: Self) -> Self {
                $self_ident(self.0.max(other.0))
            }

            /// Get the minimum of each lane.
            #[must_use]
            #[inline]
            pub fn min(self, other: Self) -> Self {
                $self_ident(self.0.min(other.0))
            }

            /// Get the reciprocal of each lane.
            #[must_use]
            #[inline]
            pub fn recip(self) -> Self {
                $self_ident(self.0.recip())
            }
        }

        #[cfg(any(feature = "std", feature = "libm"))]
        impl<$gen: num_traits::Float> $name {
            /// Get the floor of each lane.
            #[must_use]
            #[inline]
            pub fn floor(self) -> Self {
                $self_ident(self.0.floor())
            }

            /// Get the ceiling of each lane.
            #[must_use]
            #[inline]
            pub fn ceil(self) -> Self {
                $self_ident(self.0.ceil())
            }

            /// Round each lane to the nearest integer.
            #[must_use]
            #[inline]
            pub fn round(self) -> Self {
                $self_ident(self.0.round())
            }

            /// Get the square root of each lane.
            #[must_use]
            #[inline]
            pub fn sqrt(self) -> Self {
                $self_ident(self.0.sqrt())
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
