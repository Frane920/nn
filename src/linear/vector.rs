use rand::distr::uniform::SampleRange;
use rand::Rng;
use std::ops::{Add, Bound, Index, IndexMut, Mul};

#[derive(Debug, Clone)]
pub struct Vector {
    pub(crate) data: Vec<f64>,
}

impl Vector {
    pub fn new(size: usize) -> Vector {
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
            v[i] = rng.random_range(-0.5..0.5);
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
        assert_eq!(self.len(), other.len(), "Vector length mismatch");
        let mut data = Vec::with_capacity(self.data.len());
        data.extend(self.data.iter().zip(&other.data).map(|(a, b)| a + b));
        Vector { data }
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        let data = self.data.iter().map(|x| x * scalar).collect();
        Vector { data }
    }
}

impl Mul<&Vector> for &Vector {
    type Output = f64;

    fn mul(self, other: &Vector) -> f64 {
        assert_eq!(self.len(), other.len(), "Vector length mismatch");
        let mut dot = 0_f64;
        for i in 0..self.len() {
            dot += self.data[i] * other.data[i]
        }
        dot
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
