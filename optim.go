package main

import "math"

type Optimizer interface {
	Step(step, numSteps int)
}

type Adam struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Params       []*Value
	eps      float64
	mBuf         []float64
	vBuf         []float64
}

func (adam *Adam) Step(step, numSteps int) {
	lrT := adam.LearningRate * 0.5 * (1 + math.Cos(math.Pi*float64(step-1)/float64(numSteps)))
	for i, p := range adam.Params {
		adam.mBuf[i] = adam.Beta1*adam.mBuf[i] + (1-adam.Beta1)*p.Grad
		adam.vBuf[i] = adam.Beta2*adam.vBuf[i] + (1-adam.Beta2)*p.Grad*p.Grad
		mHat := adam.mBuf[i] / (1 - math.Pow(adam.Beta1, float64(step)))
		vHat := adam.vBuf[i] / (1 - math.Pow(adam.Beta2, float64(step)))
		p.Data -= lrT * mHat / (math.Sqrt(vHat) + adam.eps)
		p.Grad = 0
	}
}

func NewAdam(params []*Value, lr, beta1, beta2,eps float64) Optimizer {
	return &Adam{
		LearningRate: lr,
		Beta1:        beta1,
		Beta2:        beta2,
		Params:       params,
		eps:      1e-8,
		mBuf:         make([]float64, len(params)),
		vBuf:         make([]float64, len(params)),
	}
}
