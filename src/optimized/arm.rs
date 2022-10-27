// BSL 1.0/Apache 2.0 License

#[cfg(target_feature = "neon")]
mod neon {
    #[cfg(target_arch = "arm")]
    use core::arch::arm;
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64 as arm;

    use crate::optimized::{naive, AsDouble, AsQuad, Double, Quad};
}

#[cfg(not(target_feature = "neon"))]
mod neon {
    use crate::optimized::naive::{Double, Quad};

    pub(crate) type F32x2 = Double<f32>;
    pub(crate) type F32x4 = Quad<f32>;
    pub(crate) type U32x2 = Double<u32>;
    pub(crate) type U32x4 = Quad<u32>;
    pub(crate) type I32x2 = Double<i32>;
    pub(crate) type I32x4 = Quad<i32>;
}

pub(super) use neon::*;
