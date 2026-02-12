package main

import (
	"fmt"
	"slices"
)

type Tokenizer struct {
	VocabSize int
	BOS       int
	idToChar  []rune
	charToId  map[rune]int
}

func (tok *Tokenizer) Encode(c rune) int {
	return tok.charToId[c]
}

func (tok *Tokenizer) Decode(i int) rune {
	return tok.idToChar[i]
}

func BuildTokenizer(docs []string) Tokenizer {
	charSet := map[rune]bool{}
	for _, doc := range docs {
		for _, c := range doc {
			charSet[c] = true
		}
	}
	runeSlice := make([]rune, 0, len(charSet))
	for c := range charSet {
		runeSlice = append(runeSlice, c)
	}

	slices.Sort(runeSlice)

	uchars := runeSlice
	charToID := map[rune]int{}
	for i, c := range uchars {
		charToID[c] = i
	}
	BOS := len(uchars)
	vocabSize := len(uchars) + 1
	fmt.Printf("vocab size: %d\n", vocabSize)
	return Tokenizer{
		VocabSize: vocabSize,
		BOS:       BOS,
		idToChar:  uchars,
		charToId:  charToID,
	}
}
