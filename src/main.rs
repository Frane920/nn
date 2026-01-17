mod activation;
mod layer;
mod linear;
mod loss;
mod network;

fn main() {
    let layer_sizes = vec![2, 2, 1];
    let activations = vec![
        activation::Activation::Tanh,      // input
        activation::Activation::Tanh, // hidden 1
        activation::Activation::Sigmoid,   // output
    ];

    let inputs = [
        linear::Vector::new_from(vec![0.0, 0.0]),
        linear::Vector::new_from(vec![0.0, 1.0]),
        linear::Vector::new_from(vec![1.0, 0.0]),
        linear::Vector::new_from(vec![1.0, 1.0]),
    ];

    let targets = [
        linear::Vector::new_from(vec![0.0]),
        linear::Vector::new_from(vec![1.0]),
        linear::Vector::new_from(vec![1.0]),
        linear::Vector::new_from(vec![0.0]),
    ];

    let output_loss = loss::Loss::MSE;
    let mut nn = network::Network::new(layer_sizes, activations, output_loss);
    let lr = 0.02;
    let epochs = 1_000_000;
    let epochs_info = 100_000;
    nn.train(&inputs, &targets, lr, epochs, epochs_info);
}
