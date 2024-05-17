# Diffusion Equation solved using Physics-Informed Neural Networks (PINNs)

The diffusion equation is a partial differential equation (PDE) that describes the distribution of a quantity (e.g., heat, concentration) over space and time. Traditional numerical methods for solving PDEs can be computationally expensive and require fine discretizations. PINNs offer a powerful alternative by incorporating the physical laws described by PDEs into the training of neural networks.

This repository provides an implementation of PINNs to solve the diffusion equation. The code leverages the ability of neural networks to approximate complex functions and uses automatic differentiation to enforce the PDE constraints during training.

## Features

- Solves the one-dimensional diffusion equations
- Utilizes PyTorch for building and training neural networks
- Employs automatic differentiation to compute PDE residuals
