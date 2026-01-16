use crate::{activation, loss};
use crate::linear;
use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    pub(crate) weights: linear::Vector,
    pub(crate) bias: f64,
}
impl Neuron {
    pub fn new(in_size: usize) -> Neuron {
        let mut v = linear::Vector::new_size(in_size);
        let mut rng = rand::rng();
        let bound = (3.0 / in_size as f64).sqrt();
        for i in 0..in_size {
            v[i] = rng.random_range(-bound..bound);
        }
        Neuron {
            weights: v,
            bias: rand::rng().random_range(-0.5_f64..0.5_f64),
        }
    }

    pub fn forward(&self, ins: &linear::Vector) -> f64 {
        self.weights.dot(ins) + self.bias
    }



    pub fn train_one(
        &mut self,
        input: &linear::Vector,
        target: f64,
        lr: f64,
        activation: &activation::Activation,
    ) -> f64 {
        let z = self.weights.dot(input) + self.bias;
        let y = activation.apply(z);

        let loss = loss::loss::mse(y, target);

        let d_loss = loss::loss::d_mse(y, target);
        let d_z = activation.derivative(z);

        let d_loss_z = d_loss * d_z;

        for i in 0..self.weights.len() {
            let d_w_i = d_loss_z * input[i];
            self.weights[i] -= lr * d_w_i;
        }

        let d_loss_b = d_loss_z;
        self.bias -= lr * d_loss_b;

        loss
    }
    pub fn update_weights(&mut self, i: &linear::Vector, d: f64, lr: f64) {
        for j in 0..self.weights.len() {
            self.weights[j] -= lr * d * i[j];
        }
        self.bias -= lr *d;
    }
}
