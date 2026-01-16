use crate::{activation, linear, neuron};

pub struct Layer {
    neurons: Vec<neuron::Neuron>,
    activation: activation::Activation,
    l_in: Option<linear::Vector>,
    l_z: Option<linear::Vector>,
    l_out: Option<linear::Vector>,
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
            l_in: None,
            l_z: None,
            l_out: None,
        }
    }

    pub fn forward(&mut self, ins: &linear::Vector) -> linear::Vector {
        self.l_in = Some(ins.clone());
        let mut z = linear::Vector::new_size(self.neurons.len());
        let mut out = linear::Vector::new_size(self.neurons.len());
        for i in 0..self.neurons.len() {
            z[i] = self.neurons[i].forward(ins);
            out[i] = self.activation.apply(z[i]);
        }
        self.l_z = Some(z);
        self.l_out = Some(out.clone());
        out
    }

    pub fn backward(&mut self, grad_out: &linear::Vector, lr: f64) -> linear::Vector {
        let input = self.l_in.as_ref().expect("Must call forward first");
        let z = self.l_z.as_ref().expect("Must call forward first");

        let n_n = self.neurons.len();
        let n_in = input.len();

        let mut d = linear::Vector::new_size(n_n);
        for i in 0..n_n {
            let d_activation = self.activation.derivative(z[i]);
            d[i] = grad_out[i] * d_activation;
        }

        let mut grad_in = linear::Vector::new_size(n_in);
        for j in 0..n_in {
            let mut sum = 0.0;
            for i in 0..n_n {
                sum += self.neurons[i].weights[j] * d[i];
            }
            grad_in[j] = sum;
        }

        for i in 0..n_n {
            self.neurons[i].update_weights(input, d[i], lr);
        }

        grad_in
    }
}
