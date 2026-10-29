import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Cast raw numbers to Value objects
    def _to_value(self, other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._to_value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = self._to_value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other): # other * self
        return self * other

    def __radd__(self, other): # other + self
        return self + other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)
    # Update the __add__ method
    def __add__(self, other):
        other = self._to_value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    # Update the __mul__ method
    def __mul__(self, other):
        other = self._to_value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # Add the ReLU activation function
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # The core Backpropagation function
    def backward(self):
        topo = []
        visited = set()
        
        # Build topological order
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Base case: derivative of output with respect to itself is 1
        self.grad = 1.0
        
        # Apply chain rule in reverse topological order
        for node in reversed(topo):
            node._backward()