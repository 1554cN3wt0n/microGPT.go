package main

import "math"

// ---------------------------------------------------------------------------
// Autograd: Value
// ---------------------------------------------------------------------------

type Value struct {
	Data       float64
	Grad       float64
	children   []*Value
	localGrads []float64
}

func NewValue(data float64, children []*Value, localGrads []float64) *Value {
	return &Value{Data: data, children: children, localGrads: localGrads}
}

func V(data float64) *Value { return NewValue(data, nil, nil) }

func (a *Value) Add(b *Value) *Value {
	return NewValue(a.Data+b.Data, []*Value{a, b}, []float64{1, 1})
}

func (a *Value) Mul(b *Value) *Value {
	return NewValue(a.Data*b.Data, []*Value{a, b}, []float64{b.Data, a.Data})
}

func (a *Value) Pow(exp float64) *Value {
	return NewValue(math.Pow(a.Data, exp), []*Value{a}, []float64{exp * math.Pow(a.Data, exp-1)})
}

func (a *Value) Log() *Value {
	return NewValue(math.Log(a.Data), []*Value{a}, []float64{1.0 / a.Data})
}

func (a *Value) Exp() *Value {
	e := math.Exp(a.Data)
	return NewValue(e, []*Value{a}, []float64{e})
}

func (a *Value) Relu() *Value {
	r := math.Max(0, a.Data)
	lg := 0.0
	if a.Data > 0 {
		lg = 1.0
	}
	return NewValue(r, []*Value{a}, []float64{lg})
}

func (a *Value) Neg() *Value           { return a.MulF(-1) }
func (a *Value) Sub(b *Value) *Value   { return a.Add(b.Neg()) }
func (a *Value) Div(b *Value) *Value   { return a.Mul(b.Pow(-1)) }
func (a *Value) AddF(f float64) *Value { return a.Add(V(f)) }
func (a *Value) MulF(f float64) *Value { return a.Mul(V(f)) }
func (a *Value) DivF(f float64) *Value { return a.Mul(V(f).Pow(-1)) }

func (root *Value) Backward() {
	// Build topological order
	var topo []*Value
	visited := map[*Value]bool{}
	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.children {
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(root)

	root.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for j, child := range node.children {
			child.Grad += node.localGrads[j] * node.Grad
		}
	}
}
