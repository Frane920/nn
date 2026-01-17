const SQRT_2_OVER_PI: f64 = 0.7978845608028653558798921198687;
const SELU_LAMBDA: f64 = 1.0507009873554804934193349852946;
const SELU_ALPHA_LAMBDA: f64 = 1.6732632423543772848170429916717;

#[derive(Debug, Clone, Copy, PartialEq)]
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
    #[inline]
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Linear => x,
            Activation::ELU => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Activation::SELU => {
                SELU_LAMBDA
                    * if x > 0.0 {
                        x
                    } else {
                        SELU_ALPHA_LAMBDA * (x.exp() - 1.0)
                    }
            }
            Activation::Swish => x / (1.0 + (-x).exp()),
            Activation::GELU => {
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
            }
        }
    }

    #[inline]
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Linear => 1.0,
            Activation::ELU => {
                if x > 0.0 {
                    1.0
                } else {
                    x.exp()
                }
            }
            Activation::SELU => {
                SELU_LAMBDA
                    * if x > 0.0 {
                        1_f64
                    } else {
                        SELU_ALPHA_LAMBDA * x.exp()
                    }
            }
            Activation::Swish => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + (x * s * (1.0 - s))
            }
            Activation::GELU => {
                let x3 = x * x * x;
                let inner = SQRT_2_OVER_PI * (x + 0.044715 * x3);
                let t = inner.tanh();
                0.5 * (1.0 + t)
                    + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 0.134145 * x * x)
            }
        }
    }
}
