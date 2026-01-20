use ndarray::{Array, Dimension};

#[derive(Clone)]
pub struct Tensor<D, T>
where
    D: Dimension,
{
    pub data: Array<T, D>,
}