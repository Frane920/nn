use crate::math::tensor::Tensor;
use super::layer::Layer;
use std::ops::{AddAssign, Add, Mul};
use ndarray::{Ix1, Ix2, LinalgScalar};
use num_traits::Num;

pub struct Linear<T>
where
    T: Num,
{
    pub weights: Tensor<Ix2, T>,
    pub biases: Tensor<Ix1, T>,
}

impl<T> Linear<T>
where
    T: Num,
{
    pub fn new(w: Tensor<Ix2, T>, b: Tensor<Ix1, T>) -> Self {
        Self {
            weights: w,
            biases: b,
        }
    }
}

impl<T> Layer<Ix2, T> for Linear<T>
where
    T: Num + AddAssign + PartialOrd + Mul + Add,
    T: LinalgScalar
{
    fn forward(&self, input: &Tensor<Ix2, T>) -> Tensor<Ix2, T> {
        let weights = &self.weights.data;
        let biases = &self.biases.data;
        let i = &input.data;
        let mut z = i.dot(weights);
        z += biases;
        Tensor { data: z }
    }
}