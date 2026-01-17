use crate::linear;
use rand::Rng;
use std::ops::Mul;

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<f64>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn new_from(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Matrix { data, rows, cols }
    }

    pub fn new_random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let mut data = vec![0.0; rows * cols];
        for i in 0..(rows * cols) {
            data[i] = rng.random::<f64>()
        }
        Matrix { data, rows, cols }
    }
}

use std::ops::{Index, IndexMut};

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        assert!(row < self.rows && col < self.cols);
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        assert!(row < self.rows && col < self.cols);
        &mut self.data[row * self.cols + col]
    }
}

impl Mul<&linear::Vector> for &Matrix {
    type Output = linear::Vector;
    fn mul(self, vector: &linear::Vector) -> linear::Vector {
        assert_eq!(self.cols, vector.len());
        let mut v = vec![0_f64; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                v[i] += self.data[i * self.cols + j] * vector[j];
            }
        }
        linear::Vector { data: v }
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, matrix: &Matrix) -> Matrix {
        assert_eq!(self.cols, matrix.rows);

        let mut c = Matrix::new(self.rows, matrix.cols);

        for i in 0..self.rows {
            for j in 0..matrix.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[(i, k)] * matrix[(k, j)];
                }
                c[(i, j)] = sum;
            }
        }
        c
    }
}
