use ndarray::{Array, Dimension, LinalgScalar};
use num_traits::Num;
use crate::math::tensor::Tensor;
use crate::modules::layer::layer::Layer;

pub struct Network<T, D> where T: Num + LinalgScalar, D: Dimension {
    layers: Vec<Box<dyn Layer<D, T>>>,
}

impl<T, D> Network<T, D> where T: Num + LinalgScalar + std::ops::AddAssign + std::cmp::PartialOrd, D: Dimension + ndarray::RemoveAxis {
    pub fn new(layers: Vec<Box<dyn Layer<D, T>>>) -> Self {
        Self { layers }
    }
    pub fn push(mut self, layer: Box<dyn Layer<D, T>>) -> Self {
        self.layers.push(layer);
        self
    }
    pub fn forward(&self, input: &Tensor<D, T>) -> Tensor<D, T> {
        let mut current_output = input.clone();
        for layer in &self.layers {
            current_output = layer.forward(&current_output);
        }
        current_output
    }
}