use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    ReLU,
    Tanh,
    Linear,
    LeakyReLU,
    ELU,
    SELU,
    GELU,
    Swish,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,

            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::ELU => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Activation::SELU => 1.0507 * if x > 0.0 { x } else { 1.6733 * (x.exp() - 1.0) },
            Activation::GELU => {
                0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }
            Activation::Swish => x * (1.0 / (1.0 + (-x).exp())),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - self.apply(x).powi(2),
            Activation::Linear => 1.0,

            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::ELU => {
                if x > 0.0 {
                    1.0
                } else {
                    self.apply(x) + 1.0
                }
            }
            Activation::SELU => {
                if x > 0.0 {
                    1.0507
                } else {
                    1.0507 * 1.6733 * x.exp()
                }
            }
            Activation::GELU => {
                0.5 * (1. + ((2. / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
                    + x * (-x * x / 2.).exp() * 0.134145 * x * x / (2. * PI).sqrt()
            }
            Activation::Swish => {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig + x * sig * (1.0 - sig)
            }
        }
    }
}
