# microGPT.go

A minimalistic, dependency-free implementation of a Generative Pre-trained Transformer (GPT) model in pure Go. This project aims to provide the most atomic and understandable way to train and inference a GPT, ported from @karpathy's insightful Python original.

## Features

-   **Pure Go Implementation:** No external deep learning frameworks or libraries are used beyond standard Go packages.
-   **Custom Autograd Engine:** Includes a basic automatic differentiation (`Value` and `Backward()` method) for gradient computation.
-   **Character-level Tokenizer:** A simple tokenizer that converts characters to integer IDs and vice-versa.
-   **GPT Model Architecture:** Implements core GPT components:
    -   Token and Positional Embeddings
    -   RMS Normalization
    -   Multi-Head Self-Attention (with KV caching)
    -   Feed-Forward Neural Network (MLP block with ReLU^2 activation)
-   **Adam Optimizer:** Includes an implementation of the Adam optimizer with cosine learning rate annealing.
-   **Training and Inference Loops:**
    -   **Training:** Demonstrates a full training pipeline using a character-level language modeling task.
    -   **Inference:** Generates text samples from the trained model.
-   **Dataset Handling:** Automatically downloads a sample dataset (`names.txt` from Karpathy's makemore repository) if not present.

## Project Structure

-   `autograd.go`: Implements the automatic differentiation engine.
-   `func.go`: Contains utility functions like `Linear`, `Softmax`, `Rmsnorm`, `AddVecs`.
-   `main.go`: The entry point, orchestrating dataset preparation, tokenizer building, model initialization, training, and inference.
-   `matrix.go`: Provides helpers for creating `Matrix` types (slices of `*Value`).
-   `model.go`: Defines the GPT model's hyperparameters, `StateDict`, parameter flattening, and the `GPTForward` pass.
-   `optim.go`: Implements the `Optimizer` interface and the `Adam` optimizer.
-   `tokenizer.go`: Handles character-to-ID and ID-to-character mapping.
-   `train.go`: Contains the `Train` and `Infer` functions for the main loops.
-   `utils.go`: General utilities such as file downloading and dataset preparation.

## How to Build and Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/1554cN3wt0n/microGPT.go.git
    cd microGPT.go
    ```
2.  **Run the application:**
    ```bash
    go run .
    ```
    The application will automatically download the `input.txt` dataset if it's not present, train the model, and then print sample inferences to the console.

## Usage

Upon running `go run .`, the program will:
1. Download `input.txt` (a list of names) if it doesn't exist.
2. Build a character-level tokenizer from the dataset.
3. Initialize a small GPT model.
4. Train the model for 500 steps, printing the loss at each step.
5. Perform 20 inference steps, generating new names based on the trained model.

## Acknowledgements

This project is a vibed Go port inspired by and based on [@karpathy's "microGPT" implementation in Python](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
