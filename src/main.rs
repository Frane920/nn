mod activation;
mod layer;
mod linear;
mod neuron;
mod network;

fn main() {
    let layer = layer::Layer::new(4, 4, activation::Activation::Sigmoid);
    let data = linear::Vector::new_from(vec![1_f64, 2_f64, 3_f64, 4_f64]);
    println!("{:?}", layer.forward(&data))
}
