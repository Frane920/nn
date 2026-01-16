mod activation;
mod layer;
mod linear;
mod neuron;
mod network;
mod loss;

fn main() {
    let inputs = vec![
        linear::Vector::new_from(vec![0.0, 0.0]),
        linear::Vector::new_from(vec![0.0, 1.0]),
        linear::Vector::new_from(vec![1.0, 0.0]),
        linear::Vector::new_from(vec![1.0, 1.0]),
    ];

    let targets = vec![
        linear::Vector::new_from(vec![0.0]),
        linear::Vector::new_from(vec![1.0]),
        linear::Vector::new_from(vec![1.0]),
        linear::Vector::new_from(vec![0.0]),
    ];

    let mut nn = network::Network::new(
        vec![2, 2, 1],
        vec![
            activation::Activation::Tanh,
            activation::Activation::Sigmoid,
        ],
    );

    let lr = 0.12;
    let epochs = 10_000_000;

    nn.train(&inputs, &targets, lr, epochs, 10000000);



}