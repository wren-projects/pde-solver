# Partial differential equation solver

This package provides tools for solving partial differential equations using
Tensor Train Decomposition.

# Project Plan

## Goals

- TTD
- PDE solver (convection-diffusion equation)

## Clients

- FAV/KKY (probability)
- FAV/KFY (particles)
- FAV/KMA (playing around -- research)

## Etaps

### Framework for Constant Coefficients

- research
- TTD
- IO interface (API only?)
- user interface???

### Methods for Constant Coefficients

_We are attempting to solve specifically the equation:_

$u_t - a \nabla u - \epsilon \Delta u=f$

- research
- finite difference methods -- explicit
- finite difference methods -- implicit
- interpolation (linear, cubic...)
- CIR method for advection
- nonhomogenous equations
- tests
- comparison

### Framework for Variable Coefficients

- research
- update interface to allow for variable coefficients
- update comparison
- add non-negativity and enforce maximum principle into TTD

### Methods for Variable Coefficients

_We are attempting to solve specifically the equation:_

$u_t - \nabla \cdot (\vec{f} u) - \nabla \cdot (\mathbf{Q} \cdot \nabla u) = 0$

- research
- finite difference methods -- explicit
- finite difference methods -- implicit
- interpolation (linear, cubic...)
- CIR method for advection
- nonhomogenous equations
- tests

### BONUS

- nonlinear PDE
