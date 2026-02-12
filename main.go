// The most atomic way to train and inference a GPT in pure, dependency-free Go.
// This file is the complete algorithm.
// Everything else is just efficiency.
//
// Ported from @karpathy's Python original.

package main

import (
	"fmt"
	"math/rand"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	docs := PrepareDataset(rng)

	tok := BuildTokenizer(docs)

	// --- Model ---
	sd := InitStateDict(tok.VocabSize, rng)
	params := FlattenParams(sd)
	fmt.Printf("num params: %d\n", len(params))

	// --- Optimizer ---
	const (
		learningRate = 1e-2
		beta1        = 0.9
		beta2        = 0.95
		epsAdam      = 1e-8
	)
	optim := NewAdam(params, learningRate, beta1, beta2, epsAdam)
	
	// --- Training ---
	Train(sd, docs, tok, optim)

	// --- Inference ---
	Infer(sd, rng, tok)
}
