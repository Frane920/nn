use crate::{activation, layer};

struct Network {
    layers: Vec<layer::Layer>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>, activations: Vec<activation::Activation>) -> Network {
        let mut layers = Vec::<layer::Layer>::with_capacity(layer_sizes.len());
        for i in 0..layer_sizes.len() {
            layers[i] = layer::Layer::new(layer_sizes[i], layer_sizes[i + 1], activations[i].clone());
        }
        Network { layers }
    }
}