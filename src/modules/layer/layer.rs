use std::ops::AddAssign;
use num_traits::Num;
use ndarray::{Ix1, Ix2, Dimension, LinalgScalar, Array, RemoveAxis, Ix3, Ix4, Ix5, Ix6};
use crate::math::tensor::Tensor;

pub trait LowerDimension {
    type Output: Dimension;
}

macro_rules! impl_lower {
    ($high:ty, $low:ty) => {
        impl LowerDimension for $high {
            type Output = $low;
        }
    };
}

impl_lower!(Ix2, Ix1);
impl_lower!(Ix3, Ix2);
impl_lower!(Ix4, Ix3);
impl_lower!(Ix5, Ix4);
impl_lower!(Ix6, Ix5);

pub trait Layer<D, T>
where
    D: Dimension + RemoveAxis,
    T: Num + PartialOrd + AddAssign + LinalgScalar,
{
    // Only the behavior
    fn forward(&self, input: &Tensor<D, T>) -> Tensor<D, T>;
}

struct LayerGradient<D, T>
    where D: Dimension + LowerDimension,
{
    w_grad: Array<T, D>,
    b_grad: Array<T, D::Output>,
}