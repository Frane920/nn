use crate::linear;

#[derive(Clone, Copy)]
pub enum Loss {
    MSE,
    BCE,
}

impl Loss {
    pub fn calc(&self, pred: f64, target: f64) -> f64 {
        match self {
            Loss::MSE => (pred - target).powi(2),
            Loss::BCE => {
                let eps = 1e-7;
                let y = pred.clamp(eps, 1.0 - eps);
                -(target * y.ln() + (1.0 - target) * (1.0 - y).ln())
            }
        }
    }

    pub fn calc_vector(&self, pred: &linear::Vector, target: &linear::Vector) -> f64 {
        assert_eq!(pred.len(), target.len());
        let mut sum = 0.0;
        for i in 0..pred.len() {
            sum += self.calc(pred[i], target[i]);
        }
        sum / pred.len() as f64
    }

    pub fn grad(&self, pred: f64, target: f64) -> f64 {
        match self {
            Loss::MSE => 2.0 * (pred - target),
            Loss::BCE => pred - target,
        }
    }

    pub fn grad_vector(&self, pred: &linear::Vector, target: &linear::Vector) -> linear::Vector {
        assert_eq!(pred.len(), target.len());
        let mut grad = linear::Vector::new(pred.len());
        for i in 0..pred.len() {
            grad[i] = self.grad(pred[i], target[i]);
        }
        grad
    }
}
