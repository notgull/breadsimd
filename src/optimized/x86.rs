// BSL 1.0/Apache 2.0 License

//! Wrappers around SIMD primitives for f32, i32 and u32.

#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86;

#[cfg(target_feature = "sse2")]
mod sse {
    use super::x86;
    use crate::optimized::{naive, AsDouble, AsQuad, Double, Quad};

    use core::cmp;
    use core::fmt;
    use core::hash;

    /// An SIMD-optimizeable set of two f32 values.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct F32x2([f32; 2]);

    /// An SIMD-optimizeable set of four f32 values.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct F32x4(x86::__m128);

    /// An SIMD-optimizeable set of two i32 values.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct I32x2([i32; 2]);

    /// An SIMD-optimizeable set of four i32 values.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct I32x4(x86::__m128i);

    /// An SIMD-optimizeable set of two u32 values.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct U32x2([u32; 2]);

    /// An SIMD-optimizeable set of four u32 values.
    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub(crate) struct U32x4(x86::__m128i);

    impl F32x2 {
        pub(crate) fn to_f32x4(self) -> F32x4 {
            unsafe {
                let [a, b] = self.0;
                F32x4(x86::_mm_set_ps(0.0, 0.0, b, a))
            }
        }
    }

    impl F32x4 {
        pub(crate) fn packed_eq(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_castps_si128(x86::_mm_cmpeq_ps(self.0, other.0)) })
        }

        pub(crate) fn packed_gte(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_castps_si128(x86::_mm_cmpge_ps(self.0, other.0)) })
        }

        pub(crate) fn packed_lte(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_castps_si128(x86::_mm_cmple_ps(self.0, other.0)) })
        }

        pub(crate) fn xy(self) -> F32x2 {
            // Cast pointer to an array and read.
            F32x2(unsafe {
                let ptr = &self.0 as *const _ as *const [f32; 2];
                *ptr
            })
        }
    }

    impl U32x2 {
        pub(crate) fn to_u32x4(self) -> U32x4 {
            unsafe {
                let [a, b] = self.0;
                U32x4(x86::_mm_set_epi32(0, 0, b as i32, a as i32))
            }
        }

        pub(crate) fn all_true(self) -> bool {
            let mask = unsafe { core::mem::transmute::<_, u64>(self.0) };
            mask == !0
        }
    }

    impl U32x4 {
        pub(crate) fn all_true(self) -> bool {
            let mask = unsafe { x86::_mm_movemask_ps(x86::_mm_castsi128_ps(self.0)) };
            mask == 0b1111
        }

        pub(crate) fn xy(self) -> U32x2 {
            // Cast pointer to an array and read.
            U32x2(unsafe {
                let ptr = &self.0 as *const _ as *const [u32; 2];
                *ptr
            })
        }
    }

    impl From<naive::Double<u32>> for U32x2 {
        fn from(d: naive::Double<u32>) -> Self {
            U32x2(d.into_inner())
        }
    } 

    impl From<naive::Double<f32>> for F32x2 {
        fn from(value: naive::Double<f32>) -> Self {
            Self(value.into_inner())
        }
    }

    impl From<naive::Quad<f32>> for F32x4 {
        fn from(value: naive::Quad<f32>) -> Self {
            Self(unsafe { x86::_mm_loadu_ps(value.0.as_ptr()) })
        }
    }

    impl AsDouble<f32> for F32x2 {
        fn gen_new(array: [f32; 2]) -> Self {
            Self(array)
        }

        fn gen_into_inner(self) -> [f32; 2] {
            self.0
        }

        fn gen_splat(value: f32) -> Self {
            Self([value; 2])
        }

        fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("Double")
                .field(&self.0[0])
                .field(&self.0[1])
                .finish()
        }

        fn gen_add(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4().gen_add(other.to_f32x4()).0.xy())
        }

        fn gen_sub(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4().gen_sub(other.to_f32x4()).0.xy())
        }

        fn gen_mul(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4().gen_mul(other.to_f32x4()).0.xy())
        }

        fn gen_div(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4().gen_div(other.to_f32x4()).0.xy())
        }

        fn gen_bitand(self, other: Self) -> Double<f32> {
            unreachable!()
        }

        fn gen_bitor(self, other: Self) -> Double<f32> {
            unreachable!()
        }

        fn gen_bitxor(self, other: Self) -> Double<f32> {
            unreachable!()
        }

        fn gen_not(self) -> Double<f32> {
            unreachable!()
        }

        fn gen_index(&self, index: usize) -> &f32 {
            &self.0[index]
        }

        fn gen_index_mut(&mut self, index: usize) -> &mut f32 {
            &mut self.0[index]
        }

        fn gen_partial_eq(self, other: Self) -> bool {
            self.to_f32x4().packed_eq(other.to_f32x4()).all_true()
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            self.to_f32x4().gen_partial_ord(other.to_f32x4())
        }

        fn gen_default() -> Self {
            Self([0.0, 0.0])
        }

        fn gen_ord(self, other: Self) -> cmp::Ordering {
            unreachable!()
        }

        fn gen_hash<H: hash::Hasher>(&self, state: &mut H) {
            unreachable!()
        }
    }

    impl AsQuad<f32> for F32x4 {
        fn gen_new(array: [f32; 4]) -> Self {
            unsafe { F32x4(x86::_mm_loadu_ps(array.as_ptr())) }
        }

        fn gen_splat(value: f32) -> Self {
            unsafe { F32x4(x86::_mm_set1_ps(value)) }
        }

        fn gen_into_inner(self) -> [f32; 4] {
            unsafe {
                let mut result = [0.0; 4];
                x86::_mm_storeu_ps(result.as_mut_ptr(), self.0);
                result
            }
        }

        fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
        where
            f32: fmt::Debug,
        {
            let [a, b, c, d] = self.gen_into_inner();

            f.debug_tuple("Quad")
                .field(&a)
                .field(&b)
                .field(&c)
                .field(&d)
                .finish()
        }

        fn gen_add(self, other: Self) -> Quad<f32> {
            unsafe { Quad(F32x4(x86::_mm_add_ps(self.0, other.0))) }
        }

        fn gen_sub(self, other: Self) -> Quad<f32> {
            unsafe { Quad(F32x4(x86::_mm_sub_ps(self.0, other.0))) }
        }

        fn gen_mul(self, other: Self) -> Quad<f32> {
            unsafe { Quad(F32x4(x86::_mm_mul_ps(self.0, other.0))) }
        }

        fn gen_div(self, other: Self) -> Quad<f32> {
            unsafe { Quad(F32x4(x86::_mm_div_ps(self.0, other.0))) }
        }

        fn gen_index(&self, index: usize) -> &f32 {
            unsafe { &(&*(&self.0 as *const x86::__m128 as *const [f32; 4]))[index] }
        }

        fn gen_index_mut(&mut self, index: usize) -> &mut f32 {
            unsafe { &mut (&mut *(&mut self.0 as *mut x86::__m128 as *mut [f32; 4]))[index] }
        }

        fn gen_partial_eq(self, other: Self) -> bool {
            self.packed_eq(other).all_true()
        }

        fn gen_default() -> Self {
            Self::gen_splat(Default::default())
        }

        fn gen_bitand(self, other: Self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_bitor(self, other: Self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_bitxor(self, other: Self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_not(self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            match (
                self.packed_lte(other).all_true(),
                self.packed_gte(other).all_true(),
            ) {
                (true, true) => Some(cmp::Ordering::Equal),
                (true, false) => Some(cmp::Ordering::Less),
                (false, true) => Some(cmp::Ordering::Greater),
                (false, false) => None,
            }
        }

        fn gen_hash<H: core::hash::Hasher>(&self, state: &mut H) {
            unreachable!()
        }

        fn gen_ord(self, other: Self) -> cmp::Ordering {
            unreachable!()
        }
    }
}

#[cfg(not(target_feature = "sse2"))]
mod sse {
    use crate::optimized::naive::{Double, Quad};

    pub(crate) type F32x2 = Double<f32>;

    pub(crate) type F32x4 = Quad<f32>;
}

pub(super) use sse::*;
