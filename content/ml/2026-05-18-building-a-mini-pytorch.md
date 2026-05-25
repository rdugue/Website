---
layout: blog
title: Building a mini-Pytorch
description: Implementing a deep-learning framework with Numpy as the only dependency.
content: ml
stage: draft
date: 2026-05-18T19:33:00.000-04:00
thumbnail: /images/uploads/neural-network.png
---
## Objective
I recently wrote about a [career pivot](https://phito.dev/blog/2026-03-24-my-journey-so-far) to machine-learning. Whenever I am trying to teach myself a new skill as an engineer, incorporating it into an active project always helps to deepen that understanding. So with machine-learning, a model training framework like [Pytorch](https://pytorch.org/projects/pytorch/) or [Tensorflow](https://www.tensorflow.org/) seemed perfect. These frameworks build on core machine-learning fundamentals, and constantly evolve to encompass modern architectures like [Transformers](https://arxiv.org/html/1706.03762v7) as they are developed. My goal is to use this project to build common foundational architectures as I learn them.

## Scope 
There are of course some tradeoffs I am making to keep this realistic as one of many projects on my plate. The most prominent being the use of [Numpy](https://numpy.org/) for vectorized math that is the backbone of machine-learning. 
* **Pro:** Numpy uses compiled C code to efficiently execute the vectorized operations. So not only do we save time by not needing to write base linear algebra and calculus algorithms, it's simply faster than what we could ever write in Python.
* **Con:** The aforementioned frameworks we aim to emulate use GPU acceleration to compute vectorized operations faster. Numpy is CPU bound, and does not afford us that functionality.

Suffice to say, this library's intended use is for educational purposes. For now we are primarily concerned with the quality of the implementation logic. A future project could involve refactoring for GPU support with something like [Rust CUDA](https://rust-gpu.github.io/rust-cuda/) might be exciting!

## Architecture 
I organized the components as modules in an intuitive and hopefully extensible way:
* **Layers:** These are the core building blocks of the model. A layer can be for flattening data, an activation function, normalization, embedding, encoding, or simply a fully-connected (Dense, Linear) layer. All variations will implement the necessary base class methods for forward and back propagation.
* **Loss:** This module holds all the different algorithms for calculating loss. It will also have a base class for flexibility.
* **Optimization:** This module holds everything related to improve model accuracy. This means SGD and Adam implementations, weight initialization algorithms, along with the training loop.
* **Model:** This is where we put it all together! This takes our layers, our loss metric, and our optimization settings to create our trainable model. 

## Code
So we've done our planning like good little engineers, so let's take a look at the code! We're going to take a look at snippets from the core modules that should be enough for a basic understanding of how the framework does it's magic. For the full source code, skip to the conclusion.

### Layers 
We'll start by taking a look at the base layer class:
```python
import numpy as np

from ..optimization.initialization import Initializer, InitType, He


class Layer:
    """
    Base class for all layers in the network.
    """

    def __init__(self, name: str, initializer: Initializer) -> None:
        self.name = name
        self.cache = {}
        self.grads = {}
        self.initializer = initializer

    def forward(self, X: np.ndarray):
        raise NotImplementedError(f"Block '{self.name}' must implement forward method")

    def backward(self, dL_dZ):
        raise NotImplementedError(f"Block '{self.name}' must implement backward method")

    def copy(self):
        raise NotImplementedError(f"Block '{self.name}' must implement copy method")
```
It's pretty straight forward. Initializers can be specified for each layer, allowing for weight and bias initialization depending on the kind of regression we're doing(no Initializer snippet, but it's a pretty small module and easy to find in the codebase). Each layer is also responsible for managing a cache and gradients, to enable different optimizers to update weights and biases after backpropagation. Forward stores the input features in the cache before the linear regression and feed forward with the current weights and biases. Backward takes those features from the cache to calculate the gradients with respect to weights and biases.

And here is that implemented in the dense layer implementation:
```python
class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(self, input_size: int, output_size: int, initializer: Initializer=He()):
        super().__init__("dense", initializer)
        self.grads = {}
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights and biases
        rng = np.random.default_rng()
        if initializer.init_type == InitType.NORMAL:
            weights = rng.normal(size=(input_size, output_size))
        else:
            weights = rng.uniform(size=(input_size, output_size))
        self.W = weights * initializer.get_scale(weights)
        self.b = np.zeros(output_size)

    def forward(self, X: np.ndarray):
        """
        X: (batch_size, input_size) -> (batch_size, output_size)
        """
        self.cache["X"] = X
        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, dL_dZ):
        X = self.cache["X"]
        m = X.shape[0]  # batch size

        # Gradient w.r.t. weights: (1/m) * X^T @ dL_dZ
        self.grads["W"] = np.dot(X.T, dL_dZ) / m

        # Gradient w.r.t. bias: (1/m) * sum(dL_dZ)
        self.grads["b"] = np.sum(dL_dZ, axis=0) / m

        # Gradient w.r.t. input: dL_dZ @ W^T
        dL_dX = np.dot(dL_dZ, self.W.T)

        return dL_dX

    def copy(self):
        new_layer = Dense(self.input_size, self.output_size)
        new_layer.W = self.W.copy()
        new_layer.b = self.b.copy()
        new_layer.grads = {k: v.copy() for k, v in self.grads.items()}
        new_layer.cache = {k: v.copy() for k, v in self.cache.items()}
        return new_layer
```
Notice the backward method implementation passes the gradients with respect to inputs up the layer chain at end.

Finally let's look at the Softmax implementation:
```python
import numpy as np

from .base import Layer


class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__("softmax", None)

    def forward(self, X):
        self.cache["X"] = X
        axis = None if X.ndim < 2 else 1
        max_a = np.max(X, axis=axis, keepdims=True)

        dividend = np.exp(X - max_a)
        divisor = np.sum(np.exp(X - max_a), axis=axis, keepdims=True)

        self.cache["Z"] = dividend / divisor
        return self.cache["Z"]

    def backward(self, dL_dZ):
        return dL_dZ

    def copy(self):
        new_layer = Softmax()
        new_layer.cache = self.cache.copy()
        return new_layer
```
The main takeaway here is I am assuming Softmax activation will be used with Categorical Cross Entropy loss, so gradient of the loss is a straight pass to the gradient with respect to inputs.

### Loss
Next up we have the base loss class:
```python
import numpy as np


class LossBase:
    def __init__(self, name) -> None:
        self.name = name

    def loss_func(self, y_pred, y_true):
        raise NotImplementedError(f"{self.name} must implement the loss_func method.")

    def loss_gradient(self, y_pred, y_true):
        raise NotImplementedError(
            f"{self.name} must implement the loss_gradient method."
        )
```
Each loss class implements a method to calculate the loss and another to calculate the gradient with respect to the loss.

And here is the implementation for CCE:
```python
class CategoricalCrossEntropy(LossBase):
    def __init__(self) -> None:
        super().__init__("CategoricalCrossEntropy")

    def loss_func(self, y_pred, y_true):
        N = len(y_true)
        correct = y_pred[np.arange(N), y_true]
        return -np.mean(np.log(correct + 1e-8))

    def loss_gradient(self, y_pred, y_true):
        # Fused Softmax + CCE gradient: (y_pred - one_hot(y_true)) / N
        N = len(y_true)
        grad = y_pred.copy()
        grad[np.arange(N), y_true] -= 1.0
        return grad / N
```

### Optimization
For optimization we'll look at the optimizer base class and the Adam implementation:
```python
import numpy as np


class Optimizer:
    def __init__(self, name: str) -> None:
        self.name = name

    def step(self, layers):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__("Adam")
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if layer.grads:
                for param_name, g in layer.grads.items():
                    key = (id(layer), param_name)

                    if key not in self.m:
                        self.m[key] = np.zeros_like(g)
                        self.v[key] = np.zeros_like(g)

                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2

                    m_hat = self.m[key] / (1 - self.beta1**self.t)
                    v_hat = self.v[key] / (1 - self.beta2**self.t)

                    param = getattr(layer, param_name)
                    param -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
```
Each Optimizer implements a step method. In the case Adam, we use the gradients stored by each layer to calculate their respective first and second momentum. Those are in turn used to update their weights and biases.

And now let's look at the training loop:
```python
import numpy as np
from .optimizers import Optimizer


def train_loop(
    model, X, y, X_test, y_test, loss_class, optimizer, epochs=1000, batch_size=None
):

    losses = []
    rng = np.random.default_rng()

    for epoch in range(epochs):
        if batch_size is not None:
            for _ in range(len(X) // batch_size):
                indices = rng.integers(0, len(X), batch_size)
                X_batch = X[indices]
                y_batch = y[indices]

                # Forward pass
                y_pred = model.predict(X_batch)
                loss = loss_class.loss_func(y_pred, y_batch)

                # Compute gradient of loss w.r.t. predictions (dy)
                dy = loss_class.loss_gradient(y_pred, y_batch)

                # Backward pass with gradient of loss
                model.backward(dy)
                optimizer.step(model.layers)
        else:
            indices = rng.integers(0, len(X), len(X))
            # Forward pass
            y_pred = model.predict(X[indices])
            loss = loss_class.loss_func(y_pred, y[indices])

            # Compute gradient of loss w.r.t. predictions (dy)
            dy = loss_class.loss_gradient(y_pred, y[indices])

            # Backward pass with gradient of loss
            model.backward(dy)
            optimizer.step(model.layers)

        y_pred = model.predict(X)
        loss = loss_class.loss_func(y_pred, y)

        y_pred_test = model.predict(X_test)
        loss_test = loss_class.loss_func(y_pred_test, y_test)

        losses.append((loss, loss_test))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {loss_test:.4f}")

    return losses
```
Currently batch size controls the training mode. 

### Model 
Finally let's examine some code from our model class `Sequential`!
First our init block for reference:
```python
from . import loss as ls
from .optimization.training import train_loop
from .optimization import optimizers as o
from .optimization.initialization import He
from .layers import activation as a
from .layers import base as b


class Sequential:
    def __init__(
        self,
        *layers,
        alpha=0.01,
        optimizer: o.Optimizer=o.Adam(),
        batch_size=None,
        epochs=1000,
        loss_class=ls.MeanSquaredError(),
    ) -> None:
```
Now let's look at forward and back propagation:
```python
def predict(self, X):
    output = X
    for layer in self.layers:
        output = layer.forward(output)
    return output

def backward(self, gradient):
    # Start with gradient from loss and propagate backwards
    current_gradient = gradient

    # Iterate through layers in reverse order
    for layer in reversed(self.layers):
        # Pass gradient through layer and get gradient for previous layer
         current_gradient = layer.backward(current_gradient)
```
And the train method just calls the train function we've already went over:
```python
def train(self, X, y, X_test, y_test):
    losses = train_loop(
        model=self,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        optimizer=self.optimizer_type,
        loss_class=self.loss_class,
        batch_size=self.batch_size,
        epochs=self.epochs,
    )

    print("Training complete.")
    print("-" * 60)
    print(
        f"Starting Training Loss: {losses[0][0]:.4f} | Starting Test Loss: {losses[0][1]:.4f}"
    )
    print(
        f"Final Training Loss: {losses[-1][0]:.4f} | Final Test Loss: {losses[-1][1]:.4f}"
    )
    print(
        f"Training Loss Improvement: {losses[0][0] - losses[-1][0]:.4f} Test Loss Improvement: {losses[0][1] - losses[-1][1]:.4f}"
    )
    print("-" * 60)
    return losses
```

### Putting it all together!
Now that we've implemented the core functionality, let's look at what using the framework looks like. Here is a full example of what training the classic MNIST handwritten digits dataset looks like:
```python
import numpy as np
from datasets import load_dataset

from phitodeep.model import SequentialBuilder
from phitodeep.loss import CategoricalCrossEntropy
from phitodeep.optimization.optimizers import Adam
from phitodeep.optimization.initialization import Xavier, InitType

train_dataset = load_dataset("ylecun/mnist", split="train")
test_dataset = load_dataset("ylecun/mnist", split="test")

X_train = train_dataset["image"]
y_train = train_dataset["label"]
X_test = test_dataset["image"]
y_test = test_dataset["label"]

X_train = np.array(X_train).astype(np.float32) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test).astype(np.float32) / 255.0
y_test = np.array(y_test)
print(X_train.shape, y_train.shape)

model = (
    SequentialBuilder()
    .flatten()
    .dense(784, 128)
    .relu()
    .dense(128, 64, Xavier(InitType.NORMAL))
    .relu()
    .dense(64, 10, Xavier(InitType.NORMAL))
    .softmax()
    .optimizer(Adam())
    .loss(CategoricalCrossEntropy())
    .alpha(0.05)
    .epochs(5)
    .batch(64)
    .build()
)

model.summary()

model.train(X_train, y_train, X_test, y_test)
```
Very reminiscent of all the major frameworks.

## Conclusion 
That covers the core of building a deep learning framework. The full source code is currently hosted on [GitHub](https://github.com/PhitoDev/phito-deep). There is also [documentation](https://phito-deep.readthedocs.io/en/latest/index.html) for the project. And you can also find the package on [PyPi](https://pypi.org/project/phitodeep/) for experimenting with your own projects. As stated earlier, I plan to regularly update this project and iterate on it as I deepen my knowledge. Also look forward to more content on example use cases.
