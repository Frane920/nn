use crate::{activation, linear, neuron};

pub struct Layer {
    neurons: Vec<neuron::Neuron>,
    activation: activation::Activation,
}

impl Layer {
    pub fn new(input_size: usize, num_neurons: usize, activation: activation::Activation) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons);
        for i in 0..num_neurons {
            neurons.push(neuron::Neuron::new(input_size))
        }

        Layer {
            neurons,
            activation,
        }
    }

    pub fn forward(&self, ins: &linear::Vector) -> linear::Vector {
        let mut out = linear::Vector::new_size(self.neurons.len());
        for i in 0..self.neurons.len() {
            out[i] = self.activation.apply(self.neurons[i].forward(ins))
        }
        out
    }
}
