// BSL 1.0/Apache 2.0 License

//! Wrappers around SIMD primitives for f32, i32 and u32.

#![allow(clippy::unnecessary_operation)]

macro_rules! zip {
    (
        [$a:expr, $b:expr, $c:expr, $d:expr],
        [$e:expr, $f:expr, $g:expr, $h:expr],
        $left: ident, $right: ident,
        $usage:expr
    ) => {{
        let ($left, $right) = ($a, $e);
        $usage;
        let ($left, $right) = ($b, $f);
        $usage;
        let ($left, $right) = ($c, $g);
        $usage;
        let ($left, $right) = ($d, $h);
        $usage;
    }};
}

#[cfg(target_feature = "sse2")]
mod sse {
    #[cfg(target_arch = "x86")]
    use core::arch::x86;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as x86;

    use crate::optimized::{naive, AsDouble, AsQuad, Double, Quad};

    use core::cmp;
    use core::fmt;
    use core::hash::{self, Hash};

    const TRUE: u32 = !0;

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
        pub(crate) fn to_f32x4(self, pad: f32) -> F32x4 {
            unsafe {
                let [a, b] = self.0;
                F32x4(x86::_mm_set_ps(pad, pad, b, a))
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

    impl I32x2 {
        pub(crate) fn to_i32x4(self) -> I32x4 {
            I32x4(unsafe { x86::_mm_set_epi32(0, 0, self.0[1], self.0[0]) })
        }
    }

    impl I32x4 {
        pub(crate) fn packed_eq(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_cmpeq_epi32(self.0, other.0) })
        }

        pub(crate) fn packed_gt(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_cmpgt_epi32(self.0, other.0) })
        }

        pub(crate) fn packed_lt(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_cmplt_epi32(self.0, other.0) })
        }

        pub(crate) fn xy(self) -> I32x2 {
            // Cast pointer to an array and read.
            I32x2(unsafe {
                let ptr = &self.0 as *const _ as *const [i32; 2];
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

        pub(crate) fn packed_eq(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_cmpeq_epi32(self.0, other.0) })
        }

        pub(crate) fn packed_gt(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_cmpgt_epi32(self.0, other.0) })
        }

        pub(crate) fn packed_lt(self, other: Self) -> U32x4 {
            U32x4(unsafe { x86::_mm_cmplt_epi32(self.0, other.0) })
        }
    }

    impl From<naive::Double<i32>> for I32x2 {
        fn from(v: naive::Double<i32>) -> Self {
            Self(v.into_inner())
        }
    }

    impl From<naive::Quad<i32>> for I32x4 {
        fn from(v: naive::Quad<i32>) -> Self {
            Self::gen_new(v.into_inner())
        }
    }

    impl From<naive::Double<u32>> for U32x2 {
        fn from(d: naive::Double<u32>) -> Self {
            U32x2(d.into_inner())
        }
    }

    impl From<naive::Quad<u32>> for U32x4 {
        fn from(q: naive::Quad<u32>) -> Self {
            U32x4::gen_new(q.into_inner())
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
            Double(self.to_f32x4(0.0).gen_add(other.to_f32x4(0.0)).0.xy())
        }

        fn gen_sub(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4(0.0).gen_sub(other.to_f32x4(0.0)).0.xy())
        }

        fn gen_mul(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4(0.0).gen_mul(other.to_f32x4(0.0)).0.xy())
        }

        fn gen_div(self, other: Self) -> Double<f32> {
            Double(self.to_f32x4(1.0).gen_div(other.to_f32x4(1.0)).0.xy())
        }

        fn gen_bitand(self, _other: Self) -> Double<f32> {
            unreachable!()
        }

        fn gen_bitor(self, _other: Self) -> Double<f32> {
            unreachable!()
        }

        fn gen_bitxor(self, _other: Self) -> Double<f32> {
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
            self.to_f32x4(0.0).packed_eq(other.to_f32x4(0.0)).all_true()
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            self.to_f32x4(0.0).gen_partial_ord(other.to_f32x4(0.0))
        }

        fn gen_default() -> Self {
            Self([0.0, 0.0])
        }

        fn gen_ord(self, _other: Self) -> cmp::Ordering {
            unreachable!()
        }

        fn gen_hash<H: hash::Hasher>(&self, _state: &mut H) {
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

        fn gen_bitand(self, _other: Self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_bitor(self, _other: Self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_bitxor(self, _other: Self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_not(self) -> Quad<f32> {
            unreachable!()
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            let [a, b, c, d] = self.packed_lte(other).gen_into_inner();
            let [e, f, g, h] = self.packed_gte(other).gen_into_inner();

            zip!(
                [a, b, c, d],
                [e, f, g, h],
                left, right,
                {
                    match (left, right) {
                        (0, 0) => return None,
                        (TRUE, 0) => return Some(cmp::Ordering::Less),
                        (0, TRUE) => return Some(cmp::Ordering::Greater),
                        _ => {}
                    } 
                }
            );

            Some(cmp::Ordering::Equal)
        }

        fn gen_hash<H: core::hash::Hasher>(&self, _state: &mut H) {
            unreachable!()
        }

        fn gen_ord(self, _other: Self) -> cmp::Ordering {
            unreachable!()
        }
    }

    impl AsDouble<i32> for I32x2 {
        fn gen_new(array: [i32; 2]) -> Self {
            Self(array)
        }

        fn gen_splat(value: i32) -> Self {
            Self([value, value])
        }

        fn gen_into_inner(self) -> [i32; 2] {
            self.0
        }

        fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let [a, b] = self.gen_into_inner();

            f.debug_tuple("Double").field(&a).field(&b).finish()
        }

        fn gen_add(self, other: Self) -> Double<i32> {
            Double(self.to_i32x4().gen_add(other.to_i32x4()).0.xy())
        }

        fn gen_sub(self, other: Self) -> Double<i32> {
            Double(self.to_i32x4().gen_sub(other.to_i32x4()).0.xy())
        }

        fn gen_mul(self, other: Self) -> Double<i32> {
            Double(self.to_i32x4().gen_mul(other.to_i32x4()).0.xy())
        }

        fn gen_div(self, other: Self) -> Double<i32> {
            // There is no optimized integer division instruction in SSE2, so we
            // have to do it the slow way.
            Double::new([
                self.0[0] / other.0[0],
                self.0[1] / other.0[1],
            ])
        }

        fn gen_index(&self, index: usize) -> &i32 {
            unsafe { &(&*(&self.0 as *const [i32; 2]))[index] }
        }

        fn gen_index_mut(&mut self, index: usize) -> &mut i32 {
            unsafe { &mut (&mut *(&mut self.0 as *mut [i32; 2]))[index] }
        }

        fn gen_partial_eq(self, other: Self) -> bool {
            self.to_i32x4().gen_partial_eq(other.to_i32x4())
        }

        fn gen_default() -> Self {
            Self::gen_splat(Default::default())
        }

        fn gen_bitand(self, other: Self) -> Double<i32> {
            Double(self.to_i32x4().gen_bitand(other.to_i32x4()).0.xy())
        }

        fn gen_bitor(self, other: Self) -> Double<i32> {
            Double(self.to_i32x4().gen_bitor(other.to_i32x4()).0.xy())
        }

        fn gen_bitxor(self, other: Self) -> Double<i32> {
            Double(self.to_i32x4().gen_bitxor(other.to_i32x4()).0.xy())
        }

        fn gen_not(self) -> Double<i32> {
            Double(self.to_i32x4().gen_not().0.xy())
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            Some(self.gen_ord(other))
        }

        fn gen_hash<H: core::hash::Hasher>(&self, state: &mut H) {
            self.gen_into_inner().hash(state)
        }

        fn gen_ord(self, other: Self) -> cmp::Ordering {
            self.to_i32x4().gen_ord(other.to_i32x4())
        }
    }

    impl AsQuad<i32> for I32x4 {
        fn gen_new(array: [i32; 4]) -> Self {
            unsafe { I32x4(x86::_mm_loadu_si128(array.as_ptr() as *const x86::__m128i)) }
        }

        fn gen_splat(value: i32) -> Self {
            unsafe { I32x4(x86::_mm_set1_epi32(value)) }
        }

        fn gen_into_inner(self) -> [i32; 4] {
            unsafe {
                let mut result = [0; 4];
                x86::_mm_storeu_si128(result.as_mut_ptr() as *mut x86::__m128i, self.0);
                result
            }
        }

        fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let [a, b, c, d] = self.gen_into_inner();

            f.debug_tuple("Quad")
                .field(&a)
                .field(&b)
                .field(&c)
                .field(&d)
                .finish()
        }

        fn gen_add(self, other: Self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_add_epi32(self.0, other.0))) }
        }

        fn gen_sub(self, other: Self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_sub_epi32(self.0, other.0))) }
        }

        fn gen_mul(self, other: Self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_mullo_epi32(self.0, other.0))) }
        }

        fn gen_div(self, other: Self) -> Quad<i32> {
            // SIMD primitives for integer division are not available on x86
            let [a, b, c, d] = self.gen_into_inner();
            let [e, f, g, h] = other.gen_into_inner();

            Quad::new([a / e, b / f, c / g, d / h])
        }

        fn gen_index(&self, index: usize) -> &i32 {
            unsafe { &(&*(&self.0 as *const x86::__m128i as *const [i32; 4]))[index] }
        }

        fn gen_index_mut(&mut self, index: usize) -> &mut i32 {
            unsafe { &mut (&mut *(&mut self.0 as *mut x86::__m128i as *mut [i32; 4]))[index] }
        }

        fn gen_partial_eq(self, other: Self) -> bool {
            self.packed_eq(other).all_true()
        }

        fn gen_default() -> Self {
            Self::gen_splat(Default::default())
        }

        fn gen_bitand(self, other: Self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_and_si128(self.0, other.0))) }
        }

        fn gen_bitor(self, other: Self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_or_si128(self.0, other.0))) }
        }

        fn gen_bitxor(self, other: Self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_xor_si128(self.0, other.0))) }
        }

        fn gen_not(self) -> Quad<i32> {
            unsafe { Quad(I32x4(x86::_mm_xor_si128(self.0, x86::_mm_set1_epi32(-1)))) }
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            Some(self.gen_ord(other))
        }

        fn gen_hash<H: core::hash::Hasher>(&self, state: &mut H) {
            self.gen_into_inner().hash(state);
        }

        fn gen_ord(self, other: Self) -> cmp::Ordering {
            let [a, b, c, d] = self.packed_lt(other).gen_into_inner();
            let [e, f, g, h] = self.packed_gt(other).gen_into_inner(); 

            zip!(
                [a, b, c, d],
                [e, f, g, h],
                left, right,
                {
                    match (left, right) {
                        (TRUE, _) => return cmp::Ordering::Less,
                        (_, TRUE) => return cmp::Ordering::Greater,
                        _ => {}
                    }
                }
            );

            cmp::Ordering::Equal
        }
    }

    impl AsDouble<u32> for U32x2 {
        fn gen_new(array: [u32; 2]) -> Self {
            Self(array)
        }

        fn gen_splat(value: u32) -> Self {
            Self([value; 2])
        }

        fn gen_into_inner(self) -> [u32; 2] {
            self.0
        }

        fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let [a, b] = self.gen_into_inner();

            f.debug_tuple("Double").field(&a).field(&b).finish()
        }

        fn gen_add(self, other: Self) -> Double<u32> {
            Double(self.to_u32x4().gen_add(other.to_u32x4()).0.xy())
        }

        fn gen_sub(self, other: Self) -> Double<u32> {
            Double(self.to_u32x4().gen_sub(other.to_u32x4()).0.xy())
        }

        fn gen_mul(self, other: Self) -> Double<u32> {
            Double(self.to_u32x4().gen_mul(other.to_u32x4()).0.xy())
        }

        fn gen_div(self, other: Self) -> Double<u32> {
            // There is no SIMD primitive for integer division on x86
            let [a, b] = self.gen_into_inner();
            let [c, d] = other.gen_into_inner();

            Double::new([a / c, b / d])
        }

        fn gen_index(&self, index: usize) -> &u32 {
            &self.0[index]
        }

        fn gen_index_mut(&mut self, index: usize) -> &mut u32 {
            &mut self.0[index]
        }

        fn gen_bitand(self, other: Self) -> Double<u32> {
            Double(self.to_u32x4().gen_bitand(other.to_u32x4()).0.xy())
        }

        fn gen_bitor(self, other: Self) -> Double<u32> {
            Double(self.to_u32x4().gen_bitor(other.to_u32x4()).0.xy())
        }

        fn gen_bitxor(self, other: Self) -> Double<u32> {
            Double(self.to_u32x4().gen_bitxor(other.to_u32x4()).0.xy())
        }

        fn gen_not(self) -> Double<u32> {
            Double(self.to_u32x4().gen_not().0.xy())
        }

        fn gen_partial_eq(self, other: Self) -> bool {
            self.to_u32x4().gen_partial_eq(other.to_u32x4())
        }

        fn gen_default() -> Self {
            Self::gen_splat(Default::default())
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            self.to_u32x4().gen_partial_ord(other.to_u32x4())
        }

        fn gen_ord(self, other: Self) -> cmp::Ordering {
            self.to_u32x4().gen_ord(other.to_u32x4())
        }

        fn gen_hash<H: core::hash::Hasher>(&self, state: &mut H) {
            self.to_u32x4().gen_hash(state)
        }
    }

    impl AsQuad<u32> for U32x4 {
        fn gen_new(array: [u32; 4]) -> Self {
            unsafe { U32x4(x86::_mm_loadu_si128(array.as_ptr() as *const x86::__m128i)) }
        }

        fn gen_splat(value: u32) -> Self {
            unsafe { U32x4(x86::_mm_set1_epi32(value as i32)) }
        }

        fn gen_into_inner(self) -> [u32; 4] {
            unsafe {
                let mut result = [0; 4];
                x86::_mm_storeu_si128(result.as_mut_ptr() as *mut x86::__m128i, self.0);
                result
            }
        }

        fn gen_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let [a, b, c, d] = self.gen_into_inner();

            f.debug_tuple("Quad")
                .field(&a)
                .field(&b)
                .field(&c)
                .field(&d)
                .finish()
        }

        fn gen_add(self, other: Self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_add_epi32(self.0, other.0))) }
        }

        fn gen_sub(self, other: Self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_sub_epi32(self.0, other.0))) }
        }

        fn gen_mul(self, other: Self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_mullo_epi32(self.0, other.0))) }
        }

        fn gen_div(self, other: Self) -> Quad<u32> {
            // SIMD integer division is not provided by x86
            let [a, b, c, d] = self.gen_into_inner();
            let [e, f, g, h] = other.gen_into_inner();

            Quad(Self::gen_new([a / e, b / f, c / g, d / h]))
        }

        fn gen_bitand(self, other: Self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_and_si128(self.0, other.0))) }
        }

        fn gen_bitor(self, other: Self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_or_si128(self.0, other.0))) }
        }

        fn gen_bitxor(self, other: Self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_xor_si128(self.0, other.0))) }
        }

        fn gen_not(self) -> Quad<u32> {
            unsafe { Quad(U32x4(x86::_mm_xor_si128(self.0, x86::_mm_set1_epi32(-1)))) }
        }

        fn gen_index(&self, index: usize) -> &u32 {
            unsafe { &(&*(&self.0 as *const x86::__m128i as *const [u32; 4]))[index] }
        }

        fn gen_index_mut(&mut self, index: usize) -> &mut u32 {
            unsafe { &mut (&mut *(&mut self.0 as *mut x86::__m128i as *mut [u32; 4]))[index] }
        }

        fn gen_partial_eq(self, other: Self) -> bool {
            self.packed_eq(other).all_true()
        }

        fn gen_partial_ord(self, other: Self) -> Option<cmp::Ordering> {
            Some(self.gen_ord(other))
        }

        fn gen_ord(self, other: Self) -> cmp::Ordering {
            // NOTE: These checks may be able to be optimized.
            let [a, b, c, d] = self.packed_lt(other).gen_into_inner();
            let [e, f, g, h] = self.packed_gt(other).gen_into_inner();
            
            zip!(
                [a, b, c, d],
                [e, f, g, h],
                left, right,
                {
                    match (left, right) {
                        (TRUE, _) => return cmp::Ordering::Less,
                        (_, TRUE) => return cmp::Ordering::Greater,
                        _ => {}
                    }
                }
            );

            cmp::Ordering::Equal
        }

        fn gen_default() -> Self
        where
            u32: Default,
        {
            Self::gen_splat(Default::default())
        }

        fn gen_hash<H: hash::Hasher>(&self, state: &mut H) {
            self.gen_into_inner().hash(state)
        }
    }
}

#[cfg(not(target_feature = "sse2"))]
mod sse {
    use crate::optimized::naive::{Double, Quad};

    pub(crate) type F32x2 = Double<f32>;
    pub(crate) type F32x4 = Quad<f32>;
    pub(crate) type U32x2 = Double<u32>;
    pub(crate) type U32x4 = Quad<u32>;
    pub(crate) type I32x2 = Double<i32>;
    pub(crate) type I32x4 = Quad<i32>;
}

pub(super) use sse::*;
