"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Union

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiplies two numbers"""
    return a * b


def id(a: Union[int, float]) -> Union[int, float]:
    """Returns the input unchanged"""
    return a


def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Adds two numbers"""
    return a + b


def neg(a: Union[int, float]) -> Union[int, float]:
    """Negates a number"""
    return -a


def lt(a: Union[int, float], b: Union[int, float]) -> bool:
    """Checks if one number is less than another"""
    return a < b


def eq(a: Union[int, float], b: Union[int, float]) -> bool:
    """Checks if two numbers are equal"""
    return a == b


def max(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Returns the larger of two numbers"""
    return a if a > b else b


def is_close(
    a: Union[int, float], b: Union[int, float], tolerance: float = 1e-2
) -> bool:
    """Checks if two numbers are close in value"""
    return abs(a - b) < tolerance


def sigmoid(a: Union[int, float]) -> float:
    """Calculates the sigmoid function"""
    if a >= 0:
        return 1 / (1 + math.exp(-a))
    exp_a = math.exp(a)
    return exp_a / (1 + exp_a)


def relu(a: Union[int, float]) -> Union[int, float]:
    """Applies the ReLU activation function"""
    if a > 0:
        return a
    return 0


def log(a: Union[int, float]) -> float:
    """Calculates the natural logarithm"""
    if a <= 0:
        raise ValueError("Logarithm of non-positive values is undefined.")
    return math.log(a)


def exp(a: Union[int, float]) -> float:
    """Calculates the exponential function"""
    return math.exp(a)


def inv(a: Union[int, float]) -> float:
    """Calculates the reciprocal"""
    if a == 0:
        raise ValueError("Reciprocal of zero is undefined.")
    return 1 / a


def log_back(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Computes the derivative of log times a second arg"""
    if a <= 0:
        raise ValueError("Logarithm undefined for non-positive values.")
    return b / a


def inv_back(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Computes the derivative of reciprocal times a second arg"""
    if a == 0:
        raise ValueError("Reciprocal of zero is undefined.")
    return -b / (a**2)


def relu_back(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Computes the derivative of ReLU times a second arg"""
    if a < 0:
        return 0
    return b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable, iterable: Iterable) -> Iterable:
    """Applies the function 'func' to each element in 'iterable'.

    Args:
    ----
        func: A function to apply to each element.
        iterable: An iterable (like a list, tuple, etc.) to apply the function to.

    Returns:
    -------
        A generator with each element of 'iterable' transformed by 'func'.

    """
    for item in iterable:
        yield func(item)


def zipWith(func: Callable, iterable1: Iterable, iterable2: Iterable) -> Iterable:
    """Combines elements from two iterables using the given function 'func'.

    Args:
    ----
        func: A function that takes two arguments and returns a combined value.
        iterable1: The first iterable.
        iterable2: The second iterable.

    Returns:
    -------
        A generator that yields the combined values from 'iterable1' and 'iterable2' using 'func'.

    """
    iterator1 = iter(iterable1)
    iterator2 = iter(iterable2)

    while True:
        try:
            item1 = next(iterator1)
            item2 = next(iterator2)
            yield func(item1, item2)
        except StopIteration:
            break


def reduce(
    func: Callable, iterable: Iterable, init_val: Union[int, float] = 0.0
) -> Union[int, float]:
    """Reduces an iterable to a single value using the given function 'func'.

    Args:
    ----
        func: A function that takes two arguments and returns a single value.
        iterable: An iterable to reduce.
        init_val: Initial value of reduce

    Returns:
    -------
        The reduced single value.

    """
    iterator = iter(iterable)
    output = init_val

    for item in iterator:
        output = func(output, item)

    return output


def negList(iterable: Iterable) -> Iterable:
    """Negates all elements in a iterable.

    Args:
    ----
        iterable: A list of numbers.

    Returns:
    -------
        A list with all elements negated.

    """
    return map(lambda x: -x, iterable)


def addLists(iterable1: Iterable, iterable2: Iterable) -> Iterable:
    """Adds corresponding elements from two lists.

    Args:
    ----
        iterable1: The first list of numbers.
        iterable2: The second list of numbers.

    Returns:
    -------
        list of numbers: A new list where each element is the sum of the corresponding
        elements from `iterable1` and `iterable2`.

    """
    return zipWith(lambda x, y: add(x, y), iterable1, iterable2)


def sum(iterable: Iterable) -> Union[int, float]:
    """Sums all elements in a list.

    Args:
    ----
        iterable: A list of numbers.

    Returns:
    -------
        The sum of all elements in the list.

    """
    try:
        return reduce(lambda x, y: x + y, iterable, 0)
    except TypeError:
        return 0


def prod(iterable: Iterable) -> Union[int, float]:
    """Calculate the product of all elements in a list.

    Args:
    ----
        iterable: A list of numbers.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(lambda x, y: x * y, iterable, 1)
