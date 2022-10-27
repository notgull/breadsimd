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

use breadsimd::{Double, Quad};

fn ints_to_floats(a: [u32; 4]) -> [f32; 4] {
    [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]
}

fn run_test<T: core::fmt::Debug + Copy + PartialEq>(
    input1: [T; 4],
    input2: [T; 4],
    with_double: impl FnOnce(Double<T>, Double<T>) -> Double<T>,
    with_quad: impl FnOnce(Quad<T>, Quad<T>) -> Quad<T>,
    output: [T; 4],
) {
    // Test with Double
    let input1_double = [input1[0], input1[1]];
    let input2_double = [input2[0], input2[1]];
    let double1 = Double::new(input1_double);
    let double2 = Double::new(input2_double);

    let double_output = with_double(double1, double2);
    let double_output = double_output.into_inner();
    assert_eq!(double_output, [output[0], output[1]]);

    // Test with Quad
    let quad1 = Quad::new(input1);
    let quad2 = Quad::new(input2);

    let quad_output = with_quad(quad1, quad2);
    let quad_output = quad_output.into_inner();
    assert_eq!(quad_output, output);
}

macro_rules! run_test {
    (
        $input1:expr,
        $input2:expr,
        $with_double:expr,
        $with_quad:expr,
        $output:expr
    ) => {{
        run_test::<u128>($input1, $input2, $with_double, $with_quad, $output);
        run_test::<u32>($input1, $input2, $with_double, $with_quad, $output);
        run_test::<i32>($input1, $input2, $with_double, $with_quad, $output);
        run_test::<f32>(
            ints_to_floats($input1),
            ints_to_floats($input2),
            $with_double,
            $with_quad,
            ints_to_floats($output),
        );
    }};
    (
        no_float,
        $input1:expr,
        $input2:expr,
        $with_double:expr,
        $with_quad:expr,
        $output:expr
    ) => {{
        run_test::<u128>($input1, $input2, $with_double, $with_quad, $output);
        run_test::<u32>($input1, $input2, $with_double, $with_quad, $output);
        run_test::<i32>($input1, $input2, $with_double, $with_quad, $output);
    }};
}

#[test]
fn create() {
    run_test!(
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        |d1, _| d1,
        |q1, _| q1,
        [1, 2, 3, 4]
    );
}

#[test]
fn splat() {
    run_test!(
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        |_, _| Double::splat(1 as _),
        |_, _| Quad::splat(1 as _),
        [1, 1, 1, 1]
    );
}

#[test]
fn add() {
    run_test!(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        |d1, d2| d1 + d2,
        |q1, q2| q1 + q2,
        [6, 8, 10, 12]
    );
}

#[test]
fn sub() {
    run_test!(
        [12, 34, 56, 78],
        [9, 8, 7, 6],
        |d1, d2| d1 - d2,
        |q1, q2| q1 - q2,
        [3, 26, 49, 72]
    );
}

#[test]
fn mul() {
    run_test!(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        |d1, d2| d1 * d2,
        |q1, q2| q1 * q2,
        [5, 12, 21, 32]
    );
}

#[test]
fn int_div() {
    run_test!(
        no_float,
        [12, 34, 56, 78],
        [9, 8, 7, 6],
        |d1, d2| d1 / d2,
        |q1, q2| q1 / q2,
        [1, 4, 8, 13]
    );
}

#[test]
fn float_div() {
    run_test::<f32>(
        [12.0, 34.0, 56.0, 78.0],
        [8.0, 8.0, 7.0, 6.0],
        |d1, d2| d1 / d2,
        |q1, q2| q1 / q2,
        [1.5, 4.25, 8.0, 13.0],
    );
}

#[test]
fn bit_and() {
    run_test!(
        no_float,
        [0b1010, 0b1100, 0b1110, 0b1101],
        [0b0101, 0b0011, 0b0001, 0b0110],
        |d1, d2| d1 & d2,
        |q1, q2| q1 & q2,
        [0b0000, 0b0000, 0b0000, 0b0100]
    )
}

#[test]
fn bit_or() {
    run_test!(
        no_float,
        [0b1010, 0b1100, 0b0010, 0b1101],
        [0b0101, 0b0011, 0b0001, 0b0110],
        |d1, d2| d1 | d2,
        |q1, q2| q1 | q2,
        [0b1111, 0b1111, 0b0011, 0b1111]
    )
}

#[test]
fn bit_xor() {
    run_test!(
        no_float,
        [0b1010, 0b1100, 0b0010, 0b1101],
        [0b0101, 0b0011, 0b0001, 0b0110],
        |d1, d2| d1 ^ d2,
        |q1, q2| q1 ^ q2,
        [0b1111, 0b1111, 0b0011, 0b1011]
    )
}

#[test]
fn bit_not() {
    run_test!(
        no_float,
        [0b1010, 0b1100, 0b0010, 0b1101],
        [0, 0, 0, 0],
        |d1, _| !d1,
        |q1, _| !q1,
        [!0b1010, !0b1100, !0b0010, !0b1101]
    )
}

#[test]
fn index() {
    let mut d = Double::<i32>::new([1, 2]);
    d *= Double::splat(2);

    assert_eq!(d[0], 2);
    assert_eq!(d[1], 4);
}

#[test]
fn eq() {
    run_test!(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        |d1, d2| {
            assert_eq!(d1, d2);
            d1
        },
        |q1, q2| {
            assert_eq!(q1, q2);
            q1
        },
        [1, 2, 3, 4]
    );

    run_test!(
        [1, 4, 3, 4],
        [1, 3, 3, 5],
        |d1, d2| {
            assert_ne!(d1, d2);
            d1
        },
        |q1, q2| {
            assert_ne!(q1, q2);
            q1
        },
        [1, 4, 3, 4]
    );
}

#[test]
fn default() {
    let d = Double::<i32>::default();
    assert_eq!(d, Double::splat(0));

    let q = Quad::<i32>::default();
    assert_eq!(q, Quad::splat(0));
}

#[test]
fn ord() {
    use core::cmp;

    run_test!(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        |d1, d2| {
            assert_eq!(Some(cmp::Ordering::Equal), d1.partial_cmp(&d2));
            d1
        },
        |q1, q2| {
            assert_eq!(Some(cmp::Ordering::Equal), q1.partial_cmp(&q2));
            q1
        },
        [1, 2, 3, 4]
    );

    run_test!(
        [1, 4, 3, 4],
        [1, 3, 3, 5],
        |d1, d2| {
            assert_eq!(Some(cmp::Ordering::Greater), d1.partial_cmp(&d2));
            d1
        },
        |q1, q2| {
            assert_eq!(Some(cmp::Ordering::Greater), q1.partial_cmp(&q2));
            q1
        },
        [1, 4, 3, 4]
    );

    run_test!(
        [1, 3, 3, 5],
        [1, 4, 3, 4],
        |d1, d2| {
            assert_eq!(Some(cmp::Ordering::Less), d1.partial_cmp(&d2));
            d1
        },
        |q1, q2| {
            assert_eq!(Some(cmp::Ordering::Less), q1.partial_cmp(&q2));
            q1
        },
        [1, 3, 3, 5]
    );
}
