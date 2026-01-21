mod math;
mod modules;

use math::tensor::Tensor;
use modules::layer::Layer;
use modules::linear::Linear;
use modules::network::Network;
use modules::loss::loss::Loss;
use modules::loss::mse::MSE;
use ndarray::{Array, Ix1, Ix2};

fn main() {
    let input_data: Array<f32, Ix2> = Array::from(vec![
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]);
    let input_tensor = Tensor { data: input_data };

    let target_data: Array<f32, Ix2> = Array::from(vec![[1.0], [0.0]]);
    let target_tensor = Tensor { data: target_data };

    let weights_data: Array<f32, Ix2> = Array::from(vec![[0.5], [0.5], [0.5]]);
    let biases_data: Array<f32, Ix1> = Array::from(vec![0.1]);

    let layer = Linear::new(
        Tensor { data: weights_data },
        Tensor { data: biases_data },
    );

    let network = Network::new(vec![Box::new(layer)]);

    let prediction = network.forward(&input_tensor);

    println!("Prediction:\n{}", prediction.data);

    let mse = MSE;
    let loss_value = mse.compute(&prediction, &target_tensor);
    let gradients = mse.grad(&prediction, &target_tensor);

    println!("\nTarget:\n{}", target_tensor.data);
    println!("MSE Loss: {:.6}", loss_value);
    println!("Gradients:\n{}", gradients.data);
}