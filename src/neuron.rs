use crate::activation;
use crate::linear;
use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    weights: linear::Vector,
    bias: f64,
}
impl Neuron {
    pub fn new(in_size: usize) -> Neuron {
        Neuron {
            weights: linear::Vector::new_rand(in_size),
            bias: rand::rng().random_range(-0.5_f64..0.5_f64),
        }
    }

    pub fn forward(&self, ins: &linear::Vector) -> f64 {
        self.weights.dot(ins) + self.bias
    }

    pub fn loss(&self, prediction: f64, target: f64) -> f64 {
        (prediction - target).powi(2)
    }

    pub fn loss_derivative(&self, prediction: f64, target: f64) -> f64 {
        2_f64 * (prediction - target)
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

        let loss = self.loss(y, target);

        let d_loss = self.loss_derivative(y, target);
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
}
