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

use core::cmp;
use core::fmt;
use core::hash::{self, Hash};
use core::ops;
use core::simd::Simd;

// Use the naive primitives from stable for types that can't become SIMD vectors.
#[path = "stable.rs"]
mod naive;

/// An object that *may* be able to be converted into a SIMD vector.
trait MaybeSimd: Copy + Sized {
    /// The two-wide representation of this type.
    type Double: AsDouble<Self>;

    /// The four-wide representation of this type.
    type Quad: AsQuad<Self>;
}

impl<T: Copy> MaybeSimd for T {
    default type Double = naive::Double<T>;
    default type Quad = naive::Quad<T>;
}

macro_rules! simd_available {
    (
        $($ty:ty),* $(,)?
    ) => {
        $(
            impl MaybeSimd for $ty {
                type Double = Simd<$ty, 2>;
                type Quad = Simd<$ty, 4>;
            }
        )*
    }
}

simd_available! {
    u8, i8,
    u16, i16,
    u32, i32,
    u64, i64,
    usize, isize,
    f32, f64,
}

macro_rules! implementation {
    (
        @not_if_float
        is_float,
        $expr:expr
    ) => {
        unreachable!()
    };
    (
        @not_if_float
        not_float,
        $expr:expr
    ) => {
        $expr
    };
    (
        @simd_impl
        $len:expr,
        $struct_name:ident,
        $trait_name:ident,
    ) => {};
    (
        @simd_impl
        $len:expr,
        $struct_name:ident,
        $trait_name:ident,
        ($ty:ty,$is_float:ident)
        $($rest:tt)*
    ) => {
        impl From<naive::$struct_name<$ty>> for Simd<$ty, $len> {
            #[inline]
            fn from(v: naive::$struct_name<$ty>) -> Self {
                Self::from_array(v.into_inner())
            }
        }

        impl $trait_name<$ty> for Simd<$ty, $len> {
            fn gen_new(array: [$ty; $len]) -> Self {
                Self::from_array(array)
            }

            fn gen_splat(value: $ty) -> Self {
                Self::splat(value)
            }

            fn gen_into_inner(self) -> [$ty; $len] {
                self.to_array()
            }

            fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(self, f)
            }

            fn gen_add(self, other: Self) -> $struct_name<$ty> {
                $struct_name(self + other)
            }

            fn gen_sub(self, other: Self) -> $struct_name<$ty> {
                $struct_name(self - other)
            }

            fn gen_mul(self, other: Self) -> $struct_name<$ty> {
                $struct_name(self * other)
            }

            fn gen_div(self, other: Self) -> $struct_name<$ty> {
                $struct_name(self / other)
            }

            fn gen_bitand(self, _other: Self) -> $struct_name<$ty> {
                implementation!(
                    @not_if_float
                    $is_float,
                    $struct_name(self & _other)
                )
            }

            fn gen_bitor(self, _other: Self) -> $struct_name<$ty> {
                implementation!(
                    @not_if_float
                    $is_float,
                    $struct_name(self | _other)
                )
            }

            fn gen_bitxor(self, _other: Self) -> $struct_name<$ty> {
                implementation!(
                    @not_if_float
                    $is_float,
                    $struct_name(self ^ _other)
                )
            }

            fn gen_not(self) -> $struct_name<$ty> {
                implementation!(
                    @not_if_float
                    $is_float,
                    $struct_name(!self)
                )
            }

            fn gen_index(&self, index: usize) -> &$ty {
                &self.as_array()[index]
            }

            fn gen_index_mut(&mut self, index: usize) -> &mut $ty {
                &mut self.as_mut_array()[index]
            }

            fn gen_partial_eq(self, other: Self) -> bool {
                self == other
            }

            fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
                self.partial_cmp(&other)
            }

            fn gen_ord(self, _other: Self) -> cmp::Ordering {
                implementation!(
                    @not_if_float
                    $is_float,
                    self.cmp(&_other)
                )
            }

            fn gen_hash<H: hash::Hasher>(&self, _state: &mut H) {
                implementation!(
                    @not_if_float
                    $is_float,
                    self.hash(_state)
                )
            }

            fn gen_default() -> Self {
                Self::default()
            }
        }

        implementation! {
            @simd_impl
            $len,
            $struct_name,
            $trait_name,
            $($rest)*
        }
    };
    (
        $gen:ident,
        $len:expr,
        $struct_name:ident,
        $trait_name:ident,
        $assoc_name:ident,
    ) => {
        #[derive(Copy, Clone)]
        pub(crate) struct $struct_name<$gen: Copy>(<$gen as MaybeSimd>::$assoc_name);

        /// A trait wrapper that makes it easier to call trait functions when applicable.
        ///
        /// This is implemented by the naive wrappers as well as the SIMD wrappers. The methods
        /// are representative of the traits that are implemented on the SIMD types.
        trait $trait_name<$gen: Copy>: Copy + Sized + From<naive::$struct_name<$gen>> {
            fn gen_new(array: [$gen; $len]) -> Self;
            fn gen_splat(value: $gen) -> Self;
            fn gen_into_inner(self) -> [$gen; $len];
            fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
            where
                $gen: fmt::Debug;
            fn gen_add(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Add<Output = $gen>;
            fn gen_sub(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Sub<Output = $gen>;

            fn gen_mul(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Mul<Output = $gen>;

            fn gen_div(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Div<Output = $gen>;

            fn gen_bitand(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::BitAnd<Output = $gen>;
            fn gen_bitor(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::BitOr<Output = $gen>;

            fn gen_bitxor(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::BitXor<Output = $gen>;

            fn gen_not(self) -> $struct_name<$gen>
            where
                $gen: ops::Not<Output = $gen>;

            fn gen_index(&self, index: usize) -> &$gen;
            fn gen_index_mut(&mut self, index: usize) -> &mut $gen;
            fn gen_partial_eq(self, other: Self) -> bool
            where
                $gen: PartialEq;
            fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering>
            where
                $gen: PartialOrd;
            fn gen_ord(self, other: Self) -> cmp::Ordering
            where
                $gen: Ord;
            fn gen_hash<H: hash::Hasher>(&self, state: &mut H)
            where
                $gen: hash::Hash;
            fn gen_default() -> Self
            where
                $gen: Default;
        }

        impl<$gen: Copy> $trait_name<$gen> for naive::$struct_name<$gen> {
            #[inline]
            fn gen_new(array: [$gen; $len]) -> Self {
                naive::$struct_name::from(array)
            }

            #[inline]
            fn gen_splat(value: $gen) -> Self {
                naive::$struct_name::splat(value)
            }

            #[inline]
            fn gen_into_inner(self) -> [$gen; $len] {
                self.into_inner()
            }

            #[inline]
            fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
            where
                $gen: fmt::Debug,
            {
                fmt::Debug::fmt(self, f)
            }

            #[inline]
            fn gen_add(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Add<Output = $gen>,
            {
                $struct_name((self + other).into())
            }

            #[inline]
            fn gen_sub(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Sub<Output = $gen>,
            {
                $struct_name((self - other).into())
            }

            #[inline]
            fn gen_mul(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Mul<Output = $gen>,
            {
                $struct_name((self * other).into())
            }

            #[inline]
            fn gen_div(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::Div<Output = $gen>,
            {
                $struct_name((self / other).into())
            }

            #[inline]
            fn gen_bitand(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::BitAnd<Output = $gen>,
            {
                $struct_name((self & other).into())
            }

            #[inline]
            fn gen_bitor(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::BitOr<Output = $gen>,
            {
                $struct_name((self | other).into())
            }

            #[inline]
            fn gen_bitxor(self, other: Self) -> $struct_name<$gen>
            where
                $gen: ops::BitXor<Output = $gen>,
            {
                $struct_name((self ^ other).into())
            }

            #[inline]
            fn gen_not(self) -> $struct_name<$gen>
            where
                $gen: ops::Not<Output = $gen>,
            {
                $struct_name((!self).into())
            }

            #[inline]
            fn gen_index(&self, index: usize) -> &$gen {
                &self[index]
            }

            #[inline]
            fn gen_index_mut(&mut self, index: usize) -> &mut $gen {
                &mut self[index]
            }

            #[inline]
            fn gen_partial_eq(self, other: Self) -> bool
            where
                $gen: PartialEq,
            {
                self == other
            }

            #[inline]
            fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering>
            where
                $gen: PartialOrd,
            {
                self.partial_cmp(&other)
            }

            #[inline]
            fn gen_ord(self, other: Self) -> cmp::Ordering
            where
                $gen: Ord,
            {
                self.cmp(&other)
            }

            #[inline]
            fn gen_hash<H: hash::Hasher>(&self, state: &mut H)
            where
                $gen: hash::Hash,
            {
                hash::Hash::hash(self, state)
            }

            #[inline]
            fn gen_default() -> Self
            where
                $gen: Default,
            {
                Default::default()
            }
        }

        implementation! {
            @simd_impl
            $len,
            $struct_name,
            $trait_name,
            (i8, not_float)
            (u8, not_float)
            (i16, not_float)
            (u16, not_float)
            (i32, not_float)
            (u32, not_float)
            (i64, not_float)
            (u64, not_float)
            (isize, not_float)
            (usize, not_float)
            (f32, is_float)
            (f64, is_float)
        }

        impl<$gen: Copy> $struct_name<$gen> {
            pub(crate) fn new(array: [$gen; $len]) -> Self {
                $struct_name(<$gen as MaybeSimd>::$assoc_name::gen_new(array))
            }

            pub(crate) fn splat(value: $gen) -> Self {
                $struct_name(<$gen as MaybeSimd>::$assoc_name::gen_splat(value))
            }

            pub(crate) fn into_inner(self) -> [$gen; $len] {
                self.0.gen_into_inner()
            }
        }

        impl<$gen: Copy + fmt::Debug> fmt::Debug for $struct_name<$gen> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.gen_fmt(f)
            }
        }

        impl<$gen: Copy + ops::Add<Output = $gen>> ops::Add for $struct_name<$gen> {
            type Output = Self;

            fn add(self, other: Self) -> Self::Output {
                self.0.gen_add(other.0)
            }
        }

        impl<$gen: Copy + ops::Sub<Output = $gen>> ops::Sub for $struct_name<$gen> {
            type Output = Self;

            fn sub(self, other: Self) -> Self::Output {
                self.0.gen_sub(other.0)
            }
        }

        impl<$gen: Copy + ops::Mul<Output = $gen>> ops::Mul for $struct_name<$gen> {
            type Output = Self;

            fn mul(self, other: Self) -> Self::Output {
                self.0.gen_mul(other.0)
            }
        }

        impl<$gen: Copy + ops::Div<Output = $gen>> ops::Div for $struct_name<$gen> {
            type Output = Self;

            fn div(self, other: Self) -> Self::Output {
                self.0.gen_div(other.0)
            }
        }

        impl<$gen: Copy + ops::BitAnd<Output = $gen>> ops::BitAnd for $struct_name<$gen> {
            type Output = Self;

            fn bitand(self, other: Self) -> Self::Output {
                self.0.gen_bitand(other.0)
            }
        }

        impl<$gen: Copy + ops::BitOr<Output = $gen>> ops::BitOr for $struct_name<$gen> {
            type Output = Self;

            fn bitor(self, other: Self) -> Self::Output {
                self.0.gen_bitor(other.0)
            }
        }

        impl<$gen: Copy + ops::BitXor<Output = $gen>> ops::BitXor for $struct_name<$gen> {
            type Output = Self;

            fn bitxor(self, other: Self) -> Self::Output {
                self.0.gen_bitxor(other.0)
            }
        }

        impl<$gen: Copy + ops::Not<Output = $gen>> ops::Not for $struct_name<$gen> {
            type Output = Self;

            fn not(self) -> Self::Output {
                self.0.gen_not()
            }
        }

        impl<$gen: Copy> ops::Index<usize> for $struct_name<$gen> {
            type Output = $gen;

            fn index(&self, index: usize) -> &Self::Output {
                self.0.gen_index(index)
            }
        }

        impl<$gen: Copy> ops::IndexMut<usize> for $struct_name<$gen> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.0.gen_index_mut(index)
            }
        }

        impl<$gen: Copy + PartialEq> PartialEq for $struct_name<$gen> {
            fn eq(&self, other: &Self) -> bool {
                self.0.gen_partial_eq(other.0)
            }
        }

        impl<$gen: Copy + Eq> Eq for $struct_name<$gen> {}

        impl<$gen: Copy + PartialOrd> PartialOrd for $struct_name<$gen> {
            fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
                self.0.gen_partial_ord(other.0)
            }
        }

        impl<$gen: Copy + Ord> Ord for $struct_name<$gen> {
            fn cmp(&self, other: &Self) -> cmp::Ordering {
                self.0.gen_ord(other.0)
            }
        }

        impl<$gen: Copy + hash::Hash> hash::Hash for $struct_name<$gen> {
            fn hash<H: hash::Hasher>(&self, state: &mut H) {
                self.0.gen_hash(state)
            }
        }

        impl<$gen: Copy + Default> Default for $struct_name<$gen> {
            fn default() -> Self {
                $struct_name(<$gen as MaybeSimd>::$assoc_name::gen_default())
            }
        }

        impl<$gen: Copy> From<[$gen; $len]> for $struct_name<$gen> {
            fn from(array: [$gen; $len]) -> Self {
                $struct_name::new(array)
            }
        }
    };
}

implementation! {
    T, 2,
    Double, AsDouble, Double,
}

implementation! {
    T, 4,
    Quad, AsQuad, Quad,
}
