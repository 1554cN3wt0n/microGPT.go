package main

import (
	"fmt"
	"math/rand"
)

func Train(sd StateDict, docs []string, tok Tokenizer, optim Optimizer) {
	const numSteps = 500
	for step := 1; step <= numSteps; step++ {
		doc := docs[(step-1)%len(docs)]

		// Tokenize: BOS + char ids + BOS
		tokens := []int{tok.BOS}
		for _, c := range doc {
			tokens = append(tokens, tok.Encode(c))
		}
		tokens = append(tokens, tok.BOS)

		n := min(len(tokens)-1, blockSize)

		// Forward
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)
		for li := range keys {
			keys[li] = nil
			values[li] = nil
		}

		var losses []*Value
		for posID := range n {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := GPTForward(tokenID, posID, keys, values, sd)
			probs := Softmax(logits)
			lossT := probs[targetID].Log().Neg()
			losses = append(losses, lossT)
		}

		lossSum := losses[0]
		for _, l := range losses[1:] {
			lossSum = lossSum.Add(l)
		}
		loss := lossSum.MulF(1.0 / float64(n))

		// Backward
		loss.Backward()
		
		// Optimizer
		optim.Step(step, numSteps)

		fmt.Printf("step %4d / %4d | loss %.4f\n", step, numSteps, loss.Data)
	}
}

func Infer(sd StateDict, rng *rand.Rand, tok Tokenizer){
	const temperature = 0.5
	fmt.Println("\n--- inference ---")

	weightedChoice := func(weights []float64) int {
		total := 0.0
		for _, w := range weights {
			total += w
		}
		r := rng.Float64() * total
		cumulative := 0.0
		for i, w := range weights {
			cumulative += w
			if r <= cumulative {
				return i
			}
		}
		return len(weights) - 1
	}

	for sampleIdx := 1; sampleIdx <= 20; sampleIdx++ {
		keys := make([][][]*Value, nLayer)
		values := make([][][]*Value, nLayer)

		tokenID := tok.BOS
		var sample []rune

		for posID := 0; posID < blockSize; posID++ {
			logits := GPTForward(tokenID, posID, keys, values, sd)
			// Apply temperature
			scaled := make([]*Value, len(logits))
			for i, l := range logits {
				scaled[i] = l.MulF(1.0 / temperature)
			}
			probs := Softmax(scaled)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.Data
			}
			tokenID = weightedChoice(weights)
			if tokenID == tok.BOS {
				break
			}
			sample = append(sample, tok.Decode(tokenID))
		}

		fmt.Printf("sample %2d: %s\n", sampleIdx, string(sample))
	}
}