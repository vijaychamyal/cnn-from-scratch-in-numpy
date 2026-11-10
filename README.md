# CNN from scratch in NumPy

My implementation of Deep-ML's "Build a Trainable CNN from Scratch in NumPy" project. I worked through the layers and their gradients step by step, then put them together into a small LeNet-style network.

The code covers convolution using im2col/col2im, ReLU, max pooling, fully connected layers, softmax cross-entropy, and backpropagation. Training uses Adam; SGD is also implemented.

To run:

```bash
python -m pip install -r requirements.txt
python scaffold.py
```

The demo trains for 3 epochs on 64 synthetic 28×28 grayscale images across 3 classes. It prints the loss, train/test accuracy, and a few predictions. This is a small training demo, not an MNIST benchmark.

- `model.py` — layer implementations, gradients, optimizers, and training loop
- `scaffold.py` — dataset setup and training demo

Based on the project from Deep-ML.
