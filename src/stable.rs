// BSL 1.0/Apache 2.0 License

//! A wrapper around a tuple of values. On Stable, due to a lack of specialization, this is always just
//! a tuple.
//!
//! In certain cases, these implementations may even be auto-vectorized.

use core::cmp;
use core::fmt;
use core::hash;
use core::ops;

#[inline]
fn make_bool(b: bool) -> u32 {
    if b {
        !0
    } else {
        0
    }
}

/// A set of two values.
#[repr(transparent)]
pub(crate) struct Double<T>(pub(crate) [T; 2]);

/// A set of four values.
#[repr(transparent)]
pub(crate) struct Quad<T>(pub(crate) [T; 4]);

/// A wrapper around arrays that lets us map from one type to another.
///
/// Makes it easier to construct the macro below.
trait Foldable<T, O> {
    /// The type of the output array.
    type OutputArray;

    /// Map the array to a new array.
    fn fold(self, f: impl FnMut(T) -> O) -> Self::OutputArray;

    /// Map the array to a new array, also using elements from another array.
    fn fold2(self, other: Self, f: impl FnMut(T, T) -> O) -> Self::OutputArray;
}

impl<T, O> Foldable<T, O> for [T; 2] {
    type OutputArray = [O; 2];

    fn fold(self, mut f: impl FnMut(T) -> O) -> Self::OutputArray {
        let [a, b] = self;
        [f(a), f(b)]
    }

    fn fold2(self, other: Self, mut f: impl FnMut(T, T) -> O) -> Self::OutputArray {
        let [a, b] = self;
        let [c, d] = other;
        [f(a, c), f(b, d)]
    }
}

impl<T, O> Foldable<T, O> for [T; 4] {
    type OutputArray = [O; 4];

    fn fold(self, mut f: impl FnMut(T) -> O) -> Self::OutputArray {
        let [a, b, c, d] = self;
        [f(a), f(b), f(c), f(d)]
    }

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

        impl<$gen: Copy> Copy for $name {}

        impl<$gen: Clone> Clone for $name {
            fn clone(&self) -> Self {
                $self_ident([$(self.0[$index].clone()),*])
            }

            fn clone_from(&mut self, source: &Self) {
                $(self.0[$index].clone_from(&source.0[$index]);)*
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

        impl<$gen> $name {
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