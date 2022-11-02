# `breadsimd`

`breadsimd` is a library for SIMD (Single Instruction Multiple Data) operations on arrays of number values. The main idea with `breadsimd` is that, in Rust, it should be possible to have your cake and eat it, too. Types in `breadsimd` are generic over the type, meaning that they can contain any number value. However, using specialization, `breadsimd` can also provide optimized implementations for specific types, such as `f32` and `i64`.

If the unstable-exclusive `nightly` feature is enabled, the crate enables SIMD optimizations, through the currently-unstable `portable-simd` and `specialization` features. If not, it falls back to a naive implementation.

## Usage

```rust
use breadsimd::Double;
use core::num::Wrapping;

// At the time of writing, `Wrapping<u32>` is not optimized using SIMD.
let mut a = Double::new([Wrapping(1), Wrapping(2)]);
a += Double::new([Wrapping(3), Wrapping(4)]);
assert_eq!(a, Double::new([Wrapping(4), Wrapping(6)]));

// However, `u32` is optimized using SIMD.
let mut b = Double::<u32>::new([1, 2]);
b += Double::new([3, 4]);
assert_eq!(b, Double::new([4, 6]));
```

## Features

* `std` (enabled by default) - Enables the usage of `libstd` within `breadsimd`. This does not affect the external API, but implements `sqrt` in a more efficient way.
* `nightly` (requires a nightly compiler) - Enables certain generic types to be replaced with SIMD primitives. This is currently only supported on nightly, as it requires the `portable-simd` and `specialization` features.
* `bytemuck` - Implements `bytemuck::Pod` and `bytemuck::Zeroable` for `Double` and `Quad`.

## Dependency Justification

**Legend:** *Italics* indicate an optional dependency, while **bold** indicates a public dependency.

* ***`bytemuck`*** (version 1.12 or later) - Provides `Pod` and `Zeroable` implementations for `Double` and `Quad`. This is only enabled when the `bytemuck` feature is enabled.
* `cfg-if` (version 1.0 or later) - Provides `cfg_if!` macro, which is used to conditionally compile code based on the `nightly` feature.
* **`num-traits`** (version 0.2 or later) - Provides numerical traits like `Real` or `Signed`, which provide for generic functionality over number types.

## Future Plans

This crate will not release its 1.0.0 version until the `specialization` and `portable_simd` features in the standard library are stable.

## License

This crate is dual licensed under the Boost Software License ([`LICENSE-BOOST`]) and the Apache 2.0 License ([`LICENSE-APACHE`]), at your option. See [`LICENSE-APACHE`] and [`LICENSE-BOOST`] for details.

### Contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[`LICENSE-APACHE`]: https://github.com/notgull/breadsimd/blob/master/LICENSE-APACHE
[`LICENSE-BOOST`]: https://github.com/notgull/breadsimd/blob/master/LICENSE-BOOST
