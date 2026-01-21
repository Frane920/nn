use crate::math::tensor::Tensor;
use ndarray::Dimension;
use num_traits::Num;
use std::ops::{Add, Mul};

pub trait Loss<D, T>
where
    T: Num + Add + Mul,
    D: Dimension,
{
    fn compute(&self, prediction: &Tensor<D, T>, target: &Tensor<D, T>) -> T;
    fn grad(&self, pred: &Tensor<D, T>, target: &Tensor<D, T>) -> Tensor<D, T>;
}
