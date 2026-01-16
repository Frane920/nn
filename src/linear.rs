use rand::Rng;
use std::ops::{Add, Index, IndexMut, Mul};

#[derive(Debug, Clone)]
pub struct Vector {
    pub(crate) data: Vec<f64>,
}

impl Vector {
    pub fn new_size(size: usize) -> Vector {
        Vector {
            data: vec![0.0; size],
        }
    }

    pub fn new_from(data: Vec<f64>) -> Vector {
        Vector { data }
    }

    pub fn new_rand(size: usize) -> Vector {
        let mut rng = rand::rng();
        let mut v = Vector {
            data: vec![0.0; size],
        };
        for i in 0..v.len() {
            v[i] = rng.random_range(-1.0..1.0) / (size as f64).sqrt();
        }
        v
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }
}

impl Add for &Vector {
    type Output = Vector;
    fn add(self, other: &Vector) -> Vector {
        let mut data = Vec::with_capacity(self.data.len());
        data.extend(self.data.iter().zip(&other.data).map(|(a, b)| a + b));
        Vector { data }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        let data = self.data.into_iter()
            .map(|x| x * scalar)
            .collect();
        Vector { data }
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
