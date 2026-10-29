from nn import MLP
# Create a Multi-Layer Perceptron: 2 inputs, two hidden layers of 4, 1 output
n = MLP(2, [4, 4, 1])

# XOR Dataset
xs = [
  [2.0, 3.0, -1.0], # We only need 2 inputs for XOR, let's strictly define XOR below:
]

xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
ys = [0.0, 1.0, 1.0, 0.0] # Desired targets

# Training Loop
epochs = 500
learning_rate = 0.05

print("Starting training...")
for k in range(epochs):
    
    # 1. Forward pass
    ypred = [n(x) for x in xs]
    
    # Calculate Mean Squared Error Loss
    # sum((yout - ygt)**2)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # 2. Backward pass
    n.zero_grad()    # Flush previous gradients
    loss.backward()  # Compute new gradients via topological sort
    
    # 3. Update parameters (Stochastic Gradient Descent)
    for p in n.parameters():
        p.data -= learning_rate * p.grad
        
    if k % 50 == 0:
        print(f"Epoch {k} | Loss: {loss.data:.4f}")

print("\nFinal Predictions:")
for x, ygt in zip(xs, ys):
    ypred = n(x)
    print(f"Input: {x} | Target: {ygt} | Prediction: {ypred.data:.4f}")