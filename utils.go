package main

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
)

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func shuffleDocs(docs []string, rng *rand.Rand) {
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
}

func PrepareDataset(rng *rand.Rand) []string {
	if _, err := os.Stat("input.txt"); os.IsNotExist(err) {
		fmt.Println("Downloading dataset...")
		url := "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
		if err := downloadFile(url, "input.txt"); err != nil {
			panic("Failed to download dataset: " + err.Error())
		}
	}

	f, err := os.Open("input.txt")
	if err != nil {
		panic(err)
	}
	var docs []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	f.Close()
	shuffleDocs(docs, rng)
	fmt.Printf("num docs: %d\n", len(docs))
	return docs
}
