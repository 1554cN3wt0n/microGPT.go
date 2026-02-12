package main

import (
	"fmt"
	"math"
	"math/rand"
)

// ---------------------------------------------------------------------------
// Model hyperparameters and state
// ---------------------------------------------------------------------------

const (
	nEmbd     = 16
	nHead     = 4
	nLayer    = 1
	blockSize = 8
	headDim   = nEmbd / nHead
)

type StateDict map[string]Matrix

func InitStateDict(vocabSize int, rng *rand.Rand) StateDict {
	sd := StateDict{
		"wte":     MakeMatrix(vocabSize, nEmbd, 0.02, rng),
		"wpe":     MakeMatrix(blockSize, nEmbd, 0.02, rng),
		"lm_head": MakeMatrix(vocabSize, nEmbd, 0.02, rng),
	}
	for i := range nLayer {
		prefix := fmt.Sprintf("layer%d.", i)
		sd[prefix+"attn_wq"] = MakeMatrix(nEmbd, nEmbd, 0.02, rng)
		sd[prefix+"attn_wk"] = MakeMatrix(nEmbd, nEmbd, 0.02, rng)
		sd[prefix+"attn_wv"] = MakeMatrix(nEmbd, nEmbd, 0.02, rng)
		sd[prefix+"attn_wo"] = MakeMatrix(nEmbd, nEmbd, 0.0, rng)
		sd[prefix+"mlp_fc1"] = MakeMatrix(4*nEmbd, nEmbd, 0.02, rng)
		sd[prefix+"mlp_fc2"] = MakeMatrix(nEmbd, 4*nEmbd, 0.0, rng)
	}
	return sd
}

func FlattenParams(sd StateDict) []*Value {
	var params []*Value
	for _, mat := range sd {
		for _, row := range mat {
			params = append(params, row...)
		}
	}
	return params
}

// ---------------------------------------------------------------------------
// GPT forward pass
// keys/values are [nLayer][][]* Value — accumulated KV cache per layer
// ---------------------------------------------------------------------------

func GPTForward(tokenID, posID int, keys, values [][][]*Value, sd StateDict) []*Value {
	tokEmb := sd["wte"][tokenID]
	posEmb := sd["wpe"][posID]
	x := AddVecs(tokEmb, posEmb)
	x = Rmsnorm(x)

	for li := range nLayer {
		prefix := fmt.Sprintf("layer%d.", li)
		xResidual := x
		x = Rmsnorm(x)

		q := Linear(x, sd[prefix+"attn_wq"])
		k := Linear(x, sd[prefix+"attn_wk"])
		v := Linear(x, sd[prefix+"attn_wv"])

		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		xAttn := make([]*Value, nEmbd)
		for h := range nHead {
			hs := h * headDim
			qH := q[hs : hs+headDim]

			// Attention logits: dot(q_h, k_h[t]) / sqrt(headDim)
			attnLogits := make([]*Value, len(keys[li]))
			for t, kt := range keys[li] {
				acc := V(0)
				for j := range headDim {
					acc = acc.Add(qH[j].Mul(kt[hs+j]))
				}
				attnLogits[t] = acc.MulF(1.0 / math.Sqrt(float64(headDim)))
			}
			attnWeights := Softmax(attnLogits)

			// Weighted sum of values
			for j := range headDim {
				acc := V(0)
				for t, vt := range values[li] {
					acc = acc.Add(attnWeights[t].Mul(vt[hs+j]))
				}
				xAttn[hs+j] = acc
			}
		}

		x = Linear(xAttn, sd[prefix+"attn_wo"])
		x = AddVecs(x, xResidual)

		// MLP block
		xResidual = x
		x = Rmsnorm(x)
		x = Linear(x, sd[prefix+"mlp_fc1"])
		for i, xi := range x {
			r := xi.Relu()
			x[i] = r.Mul(r) // ReLU^2
		}
		x = Linear(x, sd[prefix+"mlp_fc2"])
		x = AddVecs(x, xResidual)
	}

	return Linear(x, sd["lm_head"])
}
