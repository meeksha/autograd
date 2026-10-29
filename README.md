# Micro-Autograd: Zero-Dependency Automatic Differentiation Engine

A lightweight, scalar-based automatic differentiation framework built entirely from scratch in Python. 

This project demonstrates the foundational mathematics and software architecture behind modern deep learning frameworks (like PyTorch and TensorFlow). By completely bypassing high-level ML libraries, this engine explicitly implements computational graph construction, topological sorting, and backpropagation via the chain rule.

## Core Architecture

* **Directed Acyclic Graph (DAG):** Mathematical operations dynamically build a DAG of `Value` objects, tracking data, gradients, and relational history.
* **Topological Sort:** Gradients are propagated backwards using a custom topological sort algorithm to ensure the chain rule is applied in the strictly correct order.
* **Zero Dependencies:** The engine relies exclusively on the Python standard library, demonstrating a deep understanding of core language mechanics and object-oriented programming (operator overloading, closures).

##  Project Structure

* `engine.py`: The core mathematical engine. Contains the `Value` scalar class, which overrides basic arithmetic operations (`+`, `*`, `**`, etc.) and implements the `backward()` method for gradient computation.
* `nn.py`: A PyTorch-esque neural network API providing `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron) abstractions to easily construct deep architectures.
* `train_xor.py`: An execution script that builds a 2-layer MLP and trains it to solve the non-linearly separable XOR problem using Stochastic Gradient Descent (SGD) and Mean Squared Error (MSE) loss.

##  Usage & Execution

Because this project has zero external dependencies, no `pip install` or virtual environment is required. 

Clone the repository and run the training script directly:

```bash
git clone [https://github.com/yourusername/micro-autograd.git](https://github.com/yourusername/micro-autograd.git)
cd micro-autograd
python train_xor.py
