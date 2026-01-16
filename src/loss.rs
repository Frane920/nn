pub mod loss {
    use crate::linear;

    pub fn mse(pred: f64, target: f64) -> f64 {
        (pred - target).powi(2)
    }

    pub fn d_mse(pred: f64, target: f64) -> f64 {
        2_f64 * (pred - target)
    }
    pub fn mse_derivative(pred: f64, target: f64) -> f64 {
        2_f64 * (pred - target)
    }

    pub fn mse_vector(pred: &linear::Vector, target: &linear::Vector) -> f64 {
        if pred.len() != target.len() {
            panic!("Prediction and target vectors must have the same length");
        }

        let mut sum = 0_f64;
        for i in 0..pred.len() {
            let d = pred[i] - target[i];
            sum += d * d;
        }

        sum / pred.len() as f64
    }



}