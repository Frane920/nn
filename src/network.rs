use crate::linear;
use crate::{activation, layer, loss};

pub struct Network {
    layers: Vec<layer::Layer>,
    output_loss: loss::Loss,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>, activations: Vec<activation::Activation>, output_loss: loss::Loss) -> Network {
        let mut layers = Vec::<layer::Layer>::with_capacity(layer_sizes.len() - 1);

        for i in 0..layer_sizes.len() - 1 {
            layers.push(layer::Layer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activations[i].clone(),
            ));
        }
        Network { layers, output_loss }
    }

    pub fn forward(&mut self, input: &linear::Vector) -> linear::Vector {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    pub fn train_one(&mut self, input: &linear::Vector, target: &linear::Vector, lr: f64) -> f64 {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        let out = current;

        let loss = self.output_loss.calc_vector(&out, target);
        let mut grad = self.output_loss.grad_vector(&out, target);

        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, lr);
        }

        loss
    }

    pub fn train(
        &mut self,
        is: &[linear::Vector],
        targets: &[linear::Vector],
        lr: f64,
        epochs: usize,
        epochs_info: usize,
    ) -> Vec<f64> {
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
