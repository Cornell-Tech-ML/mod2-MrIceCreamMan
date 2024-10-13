from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    left_args = [*vals]
    left_args[arg] -= epsilon
    left_val = f(*left_args)

    right_args = [*vals]
    right_args[arg] += epsilon
    right_val = f(*right_args)

    return (right_val - left_val) / (epsilon * 2)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate all derivatives"""
        ...

    @property
    def unique_id(self) -> int:
        """Get unique id"""
        ...

    def is_leaf(self) -> bool:
        """Check if self is leaf"""
        ...

    def is_constant(self) -> bool:
        """Check if self is constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get all parents"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply chain rule"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    sorted_list = []
    visited = set()

    def dfs(scalar: Variable) -> None:
        print(scalar.unique_id)
        if scalar.unique_id in visited:
            return

        for parent in scalar.parents:
            dfs(parent)

        visited.add(scalar.unique_id)
        sorted_list.append(scalar)

    dfs(variable)
    sorted_list.reverse()
    return sorted_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    scalar_sorted = topological_sort(variable)
    scalar_to_derivative = {}
    for scalar in scalar_sorted:
        scalar_to_derivative[scalar.unique_id] = 0
    scalar_to_derivative[variable.unique_id] = deriv

    for scalar in scalar_sorted:
        d = scalar_to_derivative[scalar.unique_id]
        if scalar.is_leaf():
            scalar.accumulate_derivative(d)
        else:
            partials = scalar.chain_rule(d)
            for var, partial in partials:
                scalar_to_derivative[var.unique_id] += partial


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns saved tensors"""
        return self.saved_values
