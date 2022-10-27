// BSL 1.0/Apache 2.0 License

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
        run_test::<u64>($input1, $input2, $with_double, $with_quad, $output);
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
fn bit_and() {

}
