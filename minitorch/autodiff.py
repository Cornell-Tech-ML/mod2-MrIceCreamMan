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

    def dfs(variable: Variable) -> None:
        if variable.unique_id in visited:
            return

        for parent in variable.parents:
            dfs(parent)

        visited.add(variable.unique_id)
        sorted_list.append(variable)

    dfs(variable)
    sorted_list.reverse()
    return sorted_list


def backpropagate(root: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        root: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    children_sorted = topological_sort(root)
    derivative_map = {}
    for child in children_sorted:
        derivative_map[child.unique_id] = 0
    derivative_map[root.unique_id] = deriv

    for child in children_sorted:
        d = derivative_map[child.unique_id]
        if child.is_leaf():
            child.accumulate_derivative(d)
        else:
            chain_rule_result = child.chain_rule(d)
            for respect_x, partial_df_dx in chain_rule_result:
                derivative_map[respect_x.unique_id] += partial_df_dx


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
