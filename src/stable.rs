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

//! A wrapper around a tuple of values. On Stable, due to a lack of specialization, this is always just
//! a tuple.
//!
//! In certain cases, these implementations may even be auto-vectorized.

#![allow(clippy::many_single_char_names)]

use core::cmp;
use core::fmt;
use core::hash;
use core::ops;

use num_traits::float::FloatCore;

/// A set of two values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Double<T>(pub(crate) [T; 2]);

/// A set of four values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Quad<T>(pub(crate) [T; 4]);

/// A wrapper around arrays that lets us map from one type to another.
///
/// Makes it easier to construct the macro below.
pub(crate) trait Foldable<T, O> {
    /// The type of the output array.
    type OutputArray;

    /// Map the array to a new array.
    fn fold(self, f: impl FnMut(T) -> O) -> Self::OutputArray;

    /// Map the array to a new array, also using elements from another array.
    fn fold2(self, other: Self, f: impl FnMut(T, T) -> O) -> Self::OutputArray;
}

impl<T, O> Foldable<T, O> for [T; 2] {
    type OutputArray = [O; 2];

    #[inline]
    fn fold(self, mut f: impl FnMut(T) -> O) -> Self::OutputArray {
        let [a, b] = self;
        [f(a), f(b)]
    }

    #[inline]
    fn fold2(self, other: Self, mut f: impl FnMut(T, T) -> O) -> Self::OutputArray {
        let [a, b] = self;
        let [c, d] = other;
        [f(a, c), f(b, d)]
    }
}

impl<T, O> Foldable<T, O> for [T; 4] {
    type OutputArray = [O; 4];

    #[inline]
    fn fold(self, mut f: impl FnMut(T) -> O) -> Self::OutputArray {
        let [a, b, c, d] = self;
        [f(a), f(b), f(c), f(d)]
    }

    #[inline]
    fn fold2(self, other: Self, mut func: impl FnMut(T, T) -> O) -> Self::OutputArray {
        let [a, b, c, d] = self;
        let [e, f, g, h] = other;
        [func(a, e), func(b, f), func(c, g), func(d, h)]
    }
}

macro_rules! implementation {
    ($gen:ident,$name:ty,$self_ident:ident,$len:expr,[$($index:literal),*]) => {
        impl<$gen: fmt::Debug> fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name))
                    $(.field(&self.0[$index]))*
                    .finish()
            }
        }

        impl<$gen: PartialEq> PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                $(self.0[$index] == other.0[$index])&&*
            }
        }

        impl<$gen: Eq> Eq for $name {}

        impl<$gen: PartialOrd> PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
                $(
                    match self.0[$index].partial_cmp(&other.0[$index]) {
                        Some(cmp::Ordering::Equal) => (),
                        non_eq => return non_eq,
                    }
                )*

                Some(cmp::Ordering::Equal)
            }
        }

        impl<$gen: Ord> Ord for $name {
            fn cmp(&self, other: &Self) -> cmp::Ordering {
                $(
                    match self.0[$index].cmp(&other.0[$index]) {
                        cmp::Ordering::Equal => (),
                        non_eq => return non_eq,
                    }
                )*

                cmp::Ordering::Equal
            }
        }

        impl<$gen: hash::Hash> hash::Hash for $name {
            fn hash<H: hash::Hasher>(&self, state: &mut H) {
                $(self.0[$index].hash(state);)*
            }
        }

        impl<$gen: ops::Add> ops::Add for $name {
            type Output = $self_ident < $gen::Output >;

            fn add(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a + b))
            }
        }

        impl<$gen: ops::Sub> ops::Sub for $name {
            type Output = $self_ident < $gen::Output >;

            fn sub(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a - b))
            }
        }

        impl<$gen: ops::Mul> ops::Mul for $name {
            type Output = $self_ident < $gen::Output >;

            fn mul(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a * b))
            }
        }

        impl<$gen: ops::Div> ops::Div for $name {
            type Output = $self_ident < $gen::Output >;

            fn div(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a / b))
            }
        }

        impl<$gen: ops::BitAnd> ops::BitAnd for $name {
            type Output = $self_ident < $gen::Output >;

            fn bitand(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a & b))
            }
        }

        impl<$gen: ops::BitOr> ops::BitOr for $name {
            type Output = $self_ident < $gen::Output >;

            fn bitor(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a | b))
            }
        }

        impl<$gen: ops::BitXor> ops::BitXor for $name {
            type Output = $self_ident < $gen::Output >;

            fn bitxor(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a ^ b))
            }
        }

        impl<$gen: ops::Not> ops::Not for $name {
            type Output = $self_ident < $gen::Output >;

            fn not(self) -> Self::Output {
                $self_ident (self.0.fold(|a| !a))
            }
        }

        impl<$gen: ops::Neg> ops::Neg for $name {
            type Output = $self_ident < $gen::Output >;

            fn neg(self) -> Self::Output {
                $self_ident (self.0.fold(|a| -a))
            }
        }

        impl<$gen: ops::Shl> ops::Shl for $name {
            type Output = $self_ident < $gen::Output >;

            fn shl(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a << b))
            }
        }

        impl<$gen: ops::Shr> ops::Shr for $name {
            type Output = $self_ident < $gen::Output >;

            fn shr(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a >> b))
            }
        }

        impl<$gen> From<[$gen; $len]> for $name {
            fn from(array: [$gen; $len]) -> Self {
                $self_ident(array)
            }
        }

        impl<$gen: Default> Default for $name {
            fn default() -> Self {
                $self_ident([$({
                    const _FOR_EACH_ITEM: &str = stringify!($index);
                    Default::default()
                }),*])
            }
        }

        impl<$gen> ops::Index<usize> for $name {
            type Output = $gen;

            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<$gen> ops::IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<$gen> AsRef<[$gen; $len]> for $name {
            fn as_ref(&self) -> &[$gen; $len] {
                &self.0
            }
        }

        impl<$gen> AsRef<[$gen]> for $name {
            fn as_ref(&self) -> &[$gen] {
                &self.0
            }
        }

        impl<$gen> AsMut<[$gen; $len]> for $name {
            fn as_mut(&mut self) -> &mut [$gen; $len] {
                &mut self.0
            }
        }

        impl<$gen> AsMut<[$gen]> for $name {
            fn as_mut(&mut self) -> &mut [$gen] {
                &mut self.0
            }
        }

        impl<$gen> $name {
            /// Create a new array.
            #[inline]
            pub(crate) fn new(array: [$gen; $len]) -> Self {
                $self_ident(array)
            }

            /// Get the underlying array.
            pub(crate) fn into_inner(self) -> [$gen; $len] {
                self.0
            }

            /// Create a new vector with one element repeated.
            pub(crate) fn splat(value: $gen) -> Self
            where
                $gen: Clone,
            {
                $self_ident([$({
                    const _FOR_EACH_ITEM: &str = stringify!($index);
                    value.clone()
                }),*])
            }
        }

        impl<$gen: FloatCore> $name {
            /// Get the absolute value of this array.
            pub(crate) fn abs(self) -> Self {
                $self_ident(self.0.fold(|a| a.abs()))
            }

            /// Find the reciprocal of this array.
            pub(crate) fn recip(self) -> Self {
                $self_ident(self.0.fold(|a| a.recip()))
            }

            /// Find the minimum of this array and another.
            pub(crate) fn min(self, other: Self) -> Self {
                $self_ident(self.0.fold2(other.0, |a, b| a.min(b)))
            }

            /// Find the maximum of this array and another.
            pub(crate) fn max(self, other: Self) -> Self {
                $self_ident(self.0.fold2(other.0, |a, b| a.max(b)))
            }
        }

        #[cfg(any(feature = "std", feature = "libm"))]
        impl<$gen: num_traits::Float> $name {
            /// Find the square root of this array.
            pub(crate) fn sqrt(self) -> Self {
                $self_ident(self.0.fold(|a| a.sqrt()))
            }

            /// Find the floor of this array.
            pub(crate) fn floor(self) -> Self {
                $self_ident(self.0.fold(|a| a.floor()))
            }

            /// Find the ceiling of this array.
            pub(crate) fn ceil(self) -> Self {
                $self_ident(self.0.fold(|a| a.ceil()))
            }

            /// Find the round of this array.
            pub(crate) fn round(self) -> Self {
                $self_ident(self.0.fold(|a| a.round()))
            }
        }
    }
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
