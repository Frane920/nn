use crate::{activation, layer, loss};
use crate::linear;

pub struct Network {
    layers: Vec<layer::Layer>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>, activations: Vec<activation::Activation>) -> Network {
        let mut layers = Vec::<layer::Layer>::with_capacity(layer_sizes.len()-1);

        for i in 0..layer_sizes.len() - 1 {
            layers.push(layer::Layer::new(layer_sizes[i], layer_sizes[i + 1], activations[i].clone()));
        }
        Network { layers }
    }

    pub fn forward(&mut self, input: &linear::Vector) -> linear::Vector {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    pub fn train_one(&mut self, i: &linear::Vector, target: &linear::Vector, lr: f64) -> f64 {
        let mut c_i = i.clone();
        for layer in &mut self.layers {
            c_i = layer.forward(&c_i);
        }
        let out = c_i;
        let loss = loss::loss::mse_vector(&out, target);
        let mut grad = linear::Vector::new_size(out.len());
        for i in 0..out.len() {
            grad[i] = 2_f64 * (out[i] - target[i]) / out.len() as f64;
        }
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, lr)
        }
        loss
    }

    pub fn train(&mut self, is: &[linear::Vector], targets: &[linear::Vector], lr: f64, epochs: usize, epochs_info: usize) -> Vec<f64> {
        let mut lossess = Vec::new();

        for epoch in 0..=epochs {
            let mut loss = 0.0;
            for (i, target) in is.iter().zip(targets.iter()) {
                loss += self.train_one(i, target, lr);
            }
            let avg_loss = loss / is.len() as f64;
            lossess.push(avg_loss);
            if epoch % epochs_info == 0 {
                println!("Epoch {}, Loss {:.16}", epoch, avg_loss)
            }
        }
        lossess

    }
}