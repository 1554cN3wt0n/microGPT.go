package main

import "math/rand"

// ---------------------------------------------------------------------------
// Matrix helpers: matrices are [][]* Value, vectors are []*Value
// ---------------------------------------------------------------------------

type Matrix [][]*Value

func MakeMatrix(nout, nin int, std float64, rng *rand.Rand) Matrix {
	m := make(Matrix, nout)
	for i := range m {
		m[i] = make([]*Value, nin)
		for j := range m[i] {
			m[i][j] = V(rng.NormFloat64() * std)
		}
	}
	return m
}
