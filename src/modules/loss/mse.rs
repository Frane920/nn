use std::ops::{Add, Mul, Sub};
use super::loss::Loss;
use crate::math::tensor::Tensor;
use ndarray::Dimension;
use num_traits::{Num, Float};

pub struct MSE;

impl<D, T> Loss<D, T> for MSE
where
    D: Dimension,
    T: Float + Num + Mul + Add + Sub + Clone + ndarray::ScalarOperand,
{
    fn compute(&self, pred: &Tensor<D, T>, target: &Tensor<D, T>) -> T {
        let diff = &pred.data - &target.data;

        let squared = &diff * &diff;

        let sum = squared.sum();
        let n = T::from(pred.data.len()).unwrap();

        sum / n
    }

    fn grad(&self, pred: &Tensor<D, T>, target: &Tensor<D, T>) -> Tensor<D, T> {
        let diff = &pred.data - &target.data;
        let n = T::from(pred.data.len()).unwrap();

        let gradient = (diff.mapv(|x| x * T::from(2).unwrap())) / n;

        Tensor { data: gradient }
    }
}