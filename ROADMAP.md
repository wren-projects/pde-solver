# Project Plan

## Goals

- Implement TTD with the full numpy interface
- Create multiple PDE solvers (methods) that use TTD for the convection-diffusion equation or its subproblems (e.g. convection equation)
- Create a basic framework to compare different solvers, their speed, memory requirements and "accuracy" of their solution (or, rather, the difference between their solution and some third solution we consider the "true" solution)

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
