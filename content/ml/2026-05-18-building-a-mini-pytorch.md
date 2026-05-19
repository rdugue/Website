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
Whenever I am trying to learn a new skill as an engineer, incorporating it into an active project always helps to deepen that understanding. So with machine-learning, a model training framework like [Pytorch]() or [Tensorflow]() seemed perfect. These frameworks build on core machine-learning fundamentals, and constantly evolve to encompass modern architectures like [Transformers]() as they are developed. My goal is to use this project to build common foundational architectures as I learn them.

## Scope 
There are of course some tradeoffs I am making to keep this realistic as one of many projects on my plate. The most prominent being the use of [Numpy]() for vectorized math that is the backbone of machine-learning. 
* **Pro:** Numpy uses compiled C code to efficiently execute the vectorized operations. So not only do we save time by not needing to write base linear algebra and calculus algorithms, it's simply faster than what we could ever write in Python.
* **Con:** The aforementioned frameworks we aim to emulate use GPU acceleration to compute vectorized operations faster. Numpy is CPU bound, and does not afford us that functionality.

Suffice to say, this library's intended use is for educational purposes. 

## Architecture 
I organized the components as modules in an intuitive and hopefully extensible way:
* **Layers:** These are the core building blocks of the model. A layer can be for flattening data, an activation function, normalization, encoding, or simply a feed forward(Dense) layer. All variations will implement the necessary base class methods for forward and back propagation.
* **Loss:** This module holds all the different algorithms for calculating loss. It will also have a base class for flexibility.
* **Optimization:**
* **Model:**

## Code

### Layers 

### Loss

### Optimization 

### Model 

## Conclusion 
