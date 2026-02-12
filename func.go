package main

func Linear(x []*Value, w Matrix) []*Value {
	out := make([]*Value, len(w))
	for i, row := range w {
		acc := V(0)
		for j, wij := range row {
			acc = acc.Add(wij.Mul(x[j]))
		}
		out[i] = acc
	}
	return out
}

func Softmax(logits []*Value) []*Value {
	maxVal := logits[0].Data
	for _, v := range logits[1:] {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}
	exps := make([]*Value, len(logits))
	for i, v := range logits {
		exps[i] = v.AddF(-maxVal).Exp()
	}
	total := exps[0]
	for _, e := range exps[1:] {
		total = total.Add(e)
	}
	out := make([]*Value, len(exps))
	for i, e := range exps {
		out[i] = e.Div(total)
	}
	return out
}

func Rmsnorm(x []*Value) []*Value {
	ms := V(0)
	for _, xi := range x {
		ms = ms.Add(xi.Mul(xi))
	}
	ms = ms.MulF(1.0 / float64(len(x)))
	scale := ms.AddF(1e-5).Pow(-0.5)
	out := make([]*Value, len(x))
	for i, xi := range x {
		out[i] = xi.Mul(scale)
	}
	return out
}

func AddVecs(a, b []*Value) []*Value {
	out := make([]*Value, len(a))
	for i := range a {
		out[i] = a[i].Add(b[i])
	}
	return out
}
