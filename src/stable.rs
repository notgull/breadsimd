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
use core::convert::TryInto;
use core::fmt;
use core::hash;
use core::marker::PhantomData;
use core::ops;

use num_traits::real::Real;
use num_traits::Signed;

/// A set of two values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Double<T: Copy>(pub(crate) [T; 2]);

/// A set of four values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct Quad<T: Copy>(pub(crate) [T; 4]);

/// A set of two boolean values for a test between two values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct DoubleMask<T> {
    pub(crate) mask: [bool; 2],
    pub(crate) phantom: PhantomData<T>,
}

/// A set of four boolean values for a test between four values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct QuadMask<T> {
    pub(crate) mask: [bool; 4],
    pub(crate) phantom: PhantomData<T>,
}

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
    ($gen:ident,$name:ty,$self_ident:ident,$len:expr,$mask_ident:ident,[$($index:literal),*]) => {
        impl<$gen: Copy> From<[bool; $len]> for $mask_ident<$gen> {
            #[inline]
            fn from(mask: [bool; $len]) -> Self {
                Self {
                    mask,
                    phantom: PhantomData,
                }
            }
        }

        impl<$gen: Copy + fmt::Debug> fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name))
                    $(.field(&self.0[$index]))*
                    .finish()
            }
        }

        impl<$gen: Copy> fmt::Debug for $mask_ident<$gen> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($mask_ident))
                    $(.field(&self.mask[$index]))*
                    .finish()
            }
        }

        impl<$gen: Copy + PartialEq> PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                $(self.0[$index] == other.0[$index])&&*
            }
        }

        impl<$gen: Copy> PartialEq for $mask_ident<$gen> {
            fn eq(&self, other: &Self) -> bool {
                $(self.mask[$index] == other.mask[$index])&&*
            }
        }

        impl<$gen: Copy + Eq> Eq for $name {}

        impl<$gen: Copy> Eq for $mask_ident<$gen> {}

        impl<$gen: Copy + PartialOrd> PartialOrd for $name {
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

        impl<$gen: Copy> PartialOrd for $mask_ident<$gen> {
            fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
                $(
                    match self.mask[$index].partial_cmp(&other.mask[$index]) {
                        Some(cmp::Ordering::Equal) => (),
                        non_eq => return non_eq,
                    }
                )*

                Some(cmp::Ordering::Equal)
            }
        }

        impl<$gen: Copy + Ord> Ord for $name {
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

        impl<$gen: Copy> Ord for $mask_ident<$gen> {
            fn cmp(&self, other: &Self) -> cmp::Ordering {
                $(
                    match self.mask[$index].cmp(&other.mask[$index]) {
                        cmp::Ordering::Equal => (),
                        non_eq => return non_eq,
                    }
                )*

                cmp::Ordering::Equal
            }
        }

        impl<$gen: Copy + hash::Hash> hash::Hash for $name {
            fn hash<H: hash::Hasher>(&self, state: &mut H) {
                $(self.0[$index].hash(state);)*
            }
        }

        impl<$gen: Copy> hash::Hash for $mask_ident<$gen> {
            fn hash<H: hash::Hasher>(&self, state: &mut H) {
                $(self.mask[$index].hash(state);)*
            }
        }

        impl<$gen: Copy + ops::Add> ops::Add for $name where <$gen as ops::Add>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn add(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a + b))
            }
        }

        impl<$gen: Copy + ops::Sub> ops::Sub for $name where <$gen as ops::Sub>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn sub(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a - b))
            }
        }

        impl<$gen: Copy + ops::Mul> ops::Mul for $name where <$gen as ops::Mul>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn mul(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a * b))
            }
        }

        impl<$gen: Copy + ops::Div> ops::Div for $name where <$gen as ops::Div>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn div(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a / b))
            }
        }

        impl<$gen: Copy + ops::BitAnd> ops::BitAnd for $name where <$gen as ops::BitAnd>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn bitand(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a & b))
            }
        }

        impl<$gen: Copy + ops::BitOr> ops::BitOr for $name where <$gen as ops::BitOr>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn bitor(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a | b))
            }
        }

        impl<$gen: Copy + ops::BitXor> ops::BitXor for $name where <$gen as ops::BitXor>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn bitxor(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a ^ b))
            }
        }

        impl<$gen: Copy + ops::Not> ops::Not for $name where <$gen as ops::Not>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn not(self) -> Self::Output {
                $self_ident (self.0.fold(|a| !a))
            }
        }

        impl<$gen: Copy> ops::BitAnd for $mask_ident<$gen> {
            type Output = $mask_ident<$gen>;

            fn bitand(self, rhs: Self) -> Self::Output {
                $mask_ident::from (self.mask.fold2(rhs.mask, |a, b| a & b))
            }
        }

        impl<$gen: Copy> ops::BitOr for $mask_ident<$gen> {
            type Output = $mask_ident<$gen>;

            fn bitor(self, rhs: Self) -> Self::Output {
                $mask_ident::from (self.mask.fold2(rhs.mask, |a, b| a | b))
            }
        }

        impl<$gen: Copy> ops::BitXor for $mask_ident<$gen> {
            type Output = $mask_ident<$gen>;

            fn bitxor(self, rhs: Self) -> Self::Output {
                $mask_ident::from (self.mask.fold2(rhs.mask, |a, b| a ^ b))
            }
        }

        impl<$gen: Copy> ops::Not for $mask_ident<$gen> {
            type Output = $mask_ident<$gen>;

            fn not(self) -> Self::Output {
                $mask_ident::from (self.mask.fold(|a| !a))
            }
        }

        impl<$gen: Copy + ops::Neg> ops::Neg for $name where <$gen as ops::Neg>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn neg(self) -> Self::Output {
                $self_ident (self.0.fold(|a| -a))
            }
        }

        impl<$gen: Copy + ops::Shl> ops::Shl for $name where <$gen as ops::Shl>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn shl(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a << b))
            }
        }

        impl<$gen: Copy + ops::Shr> ops::Shr for $name where <$gen as ops::Shr>::Output: Copy {
            type Output = $self_ident < $gen::Output >;

            fn shr(self, rhs: Self) -> Self::Output {
                $self_ident (self.0.fold2(rhs.0, |a, b| a >> b))
            }
        }

        impl<$gen: Copy> From<[$gen; $len]> for $name {
            fn from(array: [$gen; $len]) -> Self {
                $self_ident(array)
            }
        }

        impl<$gen: Copy + Default> Default for $name {
            fn default() -> Self {
                $self_ident([$({
                    const _FOR_EACH_ITEM: &str = stringify!($index);
                    Default::default()
                }),*])
            }
        }

        impl<$gen: Copy> Default for $mask_ident<$gen> {
            fn default() -> Self {
                $mask_ident::from ([$({
                    const _FOR_EACH_ITEM: &str = stringify!($index);
                    false
                }),*])
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

        impl<$gen: Copy> AsRef<[$gen; $len]> for $name {
            fn as_ref(&self) -> &[$gen; $len] {
                &self.0
            }
        }

        impl<$gen: Copy> AsRef<[$gen]> for $name {
            fn as_ref(&self) -> &[$gen] {
                &self.0
            }
        }

        impl<$gen: Copy> AsMut<[$gen; $len]> for $name {
            fn as_mut(&mut self) -> &mut [$gen; $len] {
                &mut self.0
            }
        }

        impl<$gen: Copy> AsMut<[$gen]> for $name {
            fn as_mut(&mut self) -> &mut [$gen] {
                &mut self.0
            }
        }

        impl<$gen: Copy> $name {
            /// Create a new array.
            #[inline]
            pub(crate) fn new(array: [$gen; $len]) -> Self {
                $self_ident(array)
            }

            /// Create a new array by copying it from a slice.
            #[inline]
            pub(crate) fn from_slice(slice: &[$gen]) -> Self {
                let array_ref: &[$gen; $len] =
                    &slice[..$len].try_into().expect("slice length is not equal to array length");
                $self_ident(*array_ref)
            }

            /// Get the underlying array.
            pub(crate) fn into_inner(self) -> [$gen; $len] {
                self.0
            }

            /// Create a new vector with one element repeated.
            pub(crate) fn splat(value: $gen) -> Self
            where
                $gen: Copy + Clone,
            {
                $self_ident([$({
                    const _FOR_EACH_ITEM: &str = stringify!($index);
                    value.clone()
                }),*])
            }
        }

        impl<$gen: Copy> $mask_ident<$gen> {
            /// Create a new array from a set of booleans.
            #[inline]
            pub(crate) fn from_array(array: [bool; $len]) -> Self {
                array.into()
            }

            /// Create a new array from a single element.
            #[inline]
            pub(crate) fn splat(value: bool) -> Self {
                $mask_ident::from ([$({
                    const _FOR_EACH_ITEM: &str = stringify!($index);
                    value
                }),*])
            }

            /// Convert into a set of booleans.
            #[inline]
            pub(crate) fn into_array(self) -> [bool; $len] {
                self.mask
            }

            /// Test a specific lane
            #[inline]
            pub(crate) fn test(&self, index: usize) -> bool {
                self.mask[index]
            }

            /// Set a specific lane
            #[inline]
            pub(crate) fn set(&mut self, index: usize, value: bool) {
                self.mask[index] = value;
            }

            /// Returns true if every element is true.
            #[inline]
            pub(crate) fn all(&self) -> bool {
                $(self.mask[$index] &&)* true
            }

            /// Returns true if any element is true.
            #[inline]
            pub(crate) fn any(&self) -> bool {
                $(self.mask[$index] ||)* false
            }
        }

        impl<$gen: Copy + Signed> $name {
            /// Get the absolute value of this array.
            pub(crate) fn abs(self) -> Self {
                $self_ident(self.0.fold(|a| a.abs()))
            }
        }

        impl<$gen: Copy + PartialEq> $name {
            /// Compare each element and return a mask of which elements are equal.
            pub fn packed_eq(self, other: Self) -> $mask_ident<$gen> {
                $mask_ident::from(self.0.fold2(other.0, |a, b| a == b))
            }

            /// Compare each element and return a mask of which elements are not equal.
            pub fn packed_ne(self, other: Self) -> $mask_ident<$gen> {
                $mask_ident::from(self.0.fold2(other.0, |a, b| a != b))
            }
        }

        impl<$gen: Copy + PartialOrd> $name {
            /// Compare each element and return a mask of which elements are greater than the other.
            pub fn packed_gt(self, other: Self) -> $mask_ident<$gen> {
                $mask_ident::from(self.0.fold2(other.0, |a, b| a > b))
            }

            /// Compare each element and return a mask of which elements are greater than or equal to the other.
            pub fn packed_ge(self, other: Self) -> $mask_ident<$gen> {
                $mask_ident::from(self.0.fold2(other.0, |a, b| a >= b))
            }

            /// Compare each element and return a mask of which elements are less than the other.
            pub fn packed_lt(self, other: Self) -> $mask_ident<$gen> {
                $mask_ident::from(self.0.fold2(other.0, |a, b| a < b))
            }

            /// Compare each element and return a mask of which elements are less than or equal to the other.
            pub fn packed_le(self, other: Self) -> $mask_ident<$gen> {
                $mask_ident::from(self.0.fold2(other.0, |a, b| a <= b))
            }

            /// Find the minimum of this array and another.
            pub(crate) fn min(self, other: Self) -> Self {
                $self_ident(self.0.fold2(other.0, |a, b| min(a, b)))
            }

            /// Find the maximum of this array and another.
            pub(crate) fn max(self, other: Self) -> Self {
                $self_ident(self.0.fold2(other.0, |a, b| max(a, b)))
            }

            /// Clamp this array between two other arrays.
            pub(crate) fn clamp(self, min: Self, max: Self) -> Self {
                self.max(min).min(max)
            }
        }

        impl<$gen: Copy + Real> $name {
            /// Find the reciprocal of this array.
            pub(crate) fn recip(self) -> Self {
                $self_ident(self.0.fold(|a| a.recip()))
            }

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
    DoubleMask,
    [0, 1]
}

implementation! {
    T,
    Quad<T>,
    Quad,
    4,
    QuadMask,
    [0, 1, 2, 3]
}

impl<T: Copy> Double<T> {
    /// Swap the elements of this array.
    pub(crate) fn yx(self) -> Self {
        let Self([a, b]) = self;
        Self([b, a])
    }
}

impl<T: Copy> Quad<T> {
    /// Split this `Quad` into two `Double`s.
    pub(crate) fn split(self) -> (Double<T>, Double<T>) {
        let Self([a, b, c, d]) = self;
        (Double([a, b]), Double([c, d]))
    }

    /// Get the first two elements of this array as a double.
    pub(crate) fn xy(self) -> Double<T> {
        let Self([a, b, _, _]) = self;
        Double([a, b])
    }

    /// Get the last two elements of this array as a double.
    pub(crate) fn zw(self) -> Double<T> {
        let Self([_, _, c, d]) = self;
        Double([c, d])
    }

    /// Create a new `Quad` from two `Double`s.
    pub(crate) fn from_doubles(x: Double<T>, y: Double<T>) -> Self {
        let Double([a, b]) = x;
        let Double([c, d]) = y;
        Self([a, b, c, d])
    }
}

/// PartialOrd-compatible implementation of `min`.
#[inline]
pub(crate) fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

/// PartialOrd-compatible implementation of `max`.
#[inline]
pub(crate) fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}
