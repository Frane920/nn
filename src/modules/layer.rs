use std::ops::AddAssign;
use num_traits::Num;
use ndarray::{Ix2, Dimension, LinalgScalar};
use crate::math::tensor::Tensor;

pub trait Layer<D, T>
where
    D: Dimension,
    T: Num + PartialOrd + AddAssign + LinalgScalar,
{
    // Only the behavior
    fn forward(&self, input: &Tensor<D, T>) -> Tensor<D, T>;
}