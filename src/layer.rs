use crate::{activation, linear};
use rand;
use rand::Rng;

#[derive(Debug)]
pub struct Layer {
    pub(crate) weights: linear::Matrix,
    pub(crate) biases: linear::Vector,
    pub(crate) activation: activation::Activation,
    l_in: Option<linear::Vector>,
    l_z: Option<linear::Vector>,
    l_out: Option<linear::Vector>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: activation::Activation) -> Self {
        let fan_in = input_size as f64;
        let fan_out = output_size as f64;

        let bound = match activation {
            activation::Activation::ReLU
            | activation::Activation::LeakyReLU
            | activation::Activation::GELU
            | activation::Activation::Swish
            | activation::Activation::ELU => (6.0 / fan_in).sqrt(),

            activation::Activation::SELU => (3.0 / fan_in).sqrt(),

            activation::Activation::Sigmoid
            | activation::Activation::Tanh
            | activation::Activation::Linear => (6.0 / (fan_in + fan_out)).sqrt(),
        };

        let mut weights = linear::Matrix::new(output_size, input_size);
        let mut rng = rand::rng();
        for i in 0..output_size {
            for j in 0..input_size {
                weights[(i, j)] = rng.random_range(-bound..bound);
            }
        }

        let biases = linear::Vector::new(output_size);

        Layer {
            weights,
            biases,
            activation,
            l_in: None,
            l_z: None,
            l_out: None,
        }
    }

    pub fn forward(&mut self, input: &linear::Vector) -> linear::Vector {
        self.l_in = Some(input.clone());
        let mut z = &self.weights * input;
        for i in 0..z.len() {
            z[i] += self.biases[i];
        }
        self.l_z = Some(z.clone());

        let mut out = linear::Vector::new(z.len());
        for i in 0..z.len() {
            out[i] = self.activation.apply(z[i]);
        }
        self.l_out = Some(out.clone());
        out
    }

    pub fn backward(&mut self, grad_out: &linear::Vector, lr: f64) -> linear::Vector {
        let n_out = self.weights.rows;
        let n_in = self.weights.cols;

        let input = self.l_in.as_ref().expect("forward must be called first");
        let z = self.l_z.as_ref().expect("forward must be called first");
        assert_eq!(grad_out.len(), n_out);

        let mut delta = linear::Vector::new(n_out);
        for i in 0..n_out {
            delta[i] = grad_out[i] * self.activation.derivative(z[i]);
        }

        let mut grad_in = linear::Vector::new(n_in);
        for j in 0..n_in {
            let mut sum = 0.0;
            for i in 0..n_out {
                sum += self.weights[(i, j)] * delta[i];
            }
            grad_in[j] = sum;
        }

        // update weights and biases
        for i in 0..n_out {
            for j in 0..n_in {
                self.weights[(i, j)] -= lr * delta[i] * input[j];
            }
            self.biases[i] -= lr * delta[i];
        }

        grad_in
    }
}
