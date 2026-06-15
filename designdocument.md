
# Technical Writeup

## Core Structure

The project is split into two core components:
* **TTD** — which is a custom NumPy array container,
* **Solver** — which solves a given PDE using some implementation of a NumPy
array as a state.

This modularity improves testability and allows us to implement other versions
of tensor compressions and have them work "out of the box."

While Solver, in theory, works on any `NDArray` implementation (which is selected
by the user via dependency injection), in practice, only implementations that
alter the usage of some specific methods are useful. The `NDArray` is interacted
with only in the solvers themselves (where usually
addition/subtraction/multiplication is all that is used), in `BoundaryConditions`
(where setting slices is used), and in Operators (where a few rather specific
methods are called). See [later](#API) for details.

There is also a third component, **common**, which is there strictly for the
deduplication of code across both core components, as they need to be
independent of each other.

The indetended workflow is then as follows. A user sets the equation they want
to solve and compresses their initial condition into TTD format. They feed this
to the solver and collect the solution, which they can decompress if desirable.

[diagram](docs/assets/pde.drawio.svg)


## General Design Decisions

We generally aimed for testability and scalability when making design
decisions. We required every part of the code to be type-hinted, which brought
many problems with it. For this reason, we used OOP to a great extent, even
though functional programming is perhaps better suited for this problem --- we
did not feel like there was sufficient support in python for typing a functional
code. This resulted in multiple callable classes. In the future, we will
investigate rewriting them into pure functions.

# TTD Component

This core component simply provides an alternative `NDArray` structure.

## API

TTD needs to implement all numpy array functions. The following functions,
however, are to be called on it repeatedly:
- `np.add` of two TTDs with the same dimensions
- `np.mul` with a number
- `np.tensordot`
- `np.gradient`
- `np.trace`
- `np.expand_dims`
- `np.stack`
- setting a slice of the form `[:, :, ..., :, 0, :, ..., :, :]` to 0


# Solver Component

## Overview

This core component has multiple parts. The key part is the solver itself,
which is a function (or rather a callable class) of the following type:
> `PDE : PDE → InitialCondition : NDArray → SpacialStep : Vector → BoundaryCondition : BoundaryCondition → TimeStep : DType → TargetTime : DType → FinalState : NDArray`

Where `DType` is a float (or more generally any ring) specified in pde_types
in which the whole computation is to proceed in and vector is a one dimensional
`NDArray` on the type `DType`.

## File Structure

```
packages/pde_solver
├── pyproject.toml --- configuration file of this component
├── src
│   └── pde_solver
│       ├── __init__.py
│       ├── abc --- all interfaces (ABCs) are located here
│       │   ├── __init__.py
│       │   ├── boundary.py --- ABC for boundary condition
│       │   ├── pde.py --- ABC for a PDE formulation
│       │   └── solver.py --- ABC for a solver
│       ├── boundary_conditions.py --- concrete boundary condition implementation
│       ├── operators.py --- concrete operators implementations
│       ├── pde.py --- concrete PDE formulations
│       ├── pde_types.py --- helper with type definitions
│       ├── py.typed --- marks the package as "type-hinted" for type-checkers
│       └── solvers --- concrete solver implementations 
│           ├── __init__.py
│           └── finite_differences.py
└── tests_solver --- tests
```

### PDE (Dataclass)

`PDE` describes the equation that the solver is supposed to solve. Importantly,
there are many different `PDE`s, and a solver need not be able to solve all of
them. The general notion here is that each `PDE` can be generalized in some way
there are many different kinds of PDEs, and a solver may not be able to solve all of
them (it's not a requirement). The general notion here is that each PDE can be generalized in some way
can solve some PDEs and all the more concrete versions of these `PDE`s. That is,
indeed, a lattice structure where, given two elements A and B, we define $A
\leq B$ if and only if B is a generalization of A. We leave it as an exercise
to the reader that this does follow all the lattice axioms. Now that we have a
structure for the `PDE`s, we allow each solver to state which `PDE`s it can solve
(see the Solver section for more information on this), and we can easily check
if a given `PDE` is solvable by the solver.

When it comes to actually implementing this structure, there is one problem.
Given N ways to generalize the `PDE`, there are exponentially many possible `PDE`s.
And we need all of them. Luckily, our N is reasonably small. Still, it is very
cumbersome to write out all the possibilities by hand. We have therefore opted
for a script that takes the ways of generalization as input and creates a file
with all the `PDE`s. It turns out auto-generating code with no warnings is a
non-trivial task, however, hence much time was spent there. In exchange, this
part of the code is now easily modifiable.

The way this script is executed deserves its own mention. The file itself is
necessary for type checkers to work correctly, hence it needs to be locally
available and cannot be created for runtime only. We opted for
[Just](https://just.systems/) as a shorthand for manually generating the file.
However, calling the script locally
after every push is tedious, so we decided to include the file in the remote
repository too. To ensure consistency, our pipelines run the script on every
merge request, and if a change is detected, they deny the merge request.

### BoundaryCondition (Class)

The goal of a boundary condition is to alter the `NDArray` representing the
current state in a way that makes it consistent with the theoretical condition.
This is usually done by increasing the size of the state by one in all
directions and manipulating only these boundaries. Before the solver
terminates, these boundaries are then removed.

### Solver (Callable class)

As mentioned above, the solver is at the core of this component. It takes a few
arguments, but more importantly, it knows which `PDE`s it can solve. This is done
via generic types. A parent has a generic `T`, for which each child substitutes
the `PDE`s it can solve (either the PDE type or a union of multiple `PDE`s). Via a
magic construction, the parent then intercepts the `T` each of its subclasses
selected and saves it in a dictionary. This dictionary will later be used to
select the correct solver for a given `PDE`.

As all solvers are iterative, we have abstracted the code they would share into
their parent too. Each specific `Solver` then contains only the code for one
iteration. This code should refrain from accessing the more complex methods of
the `NDArray`. Instead, it is to use operators.

### Operators (Callable class)

Operators provide an interface for solvers to compute differentials on an
`NDArray`. They are there to ensure all solvers are using the correct (optimized)
methods of the specific `NDArray`.
