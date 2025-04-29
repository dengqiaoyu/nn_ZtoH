from __future__ import annotations
from typing import Any, Callable, Final, Tuple
import math


class Value:
    """ Stores a single scalar value and its gradient """

    def __init__(self, data: Any, _children: Tuple[Value, ...] = (),
                 _op: str = "", label: str = "") -> None:
        self._data: Any = data
        self._label: str = label
        self._grad: Any = 0
        # internal variables used for autograd graph construction
        self._backward: Callable[..., Any] = lambda: None
        self._prev: set[Value] = set(_children)
        # the op that produced this node, for graphviz / debugging / etc
        self._op: str = _op

    @property
    def data(self) -> Any:
        return self._data

    @property
    def label(self) -> str:
        return self._label

    @property
    def grad(self) -> Any:
        return self._grad

    @property
    def prev(self) -> set[Value]:
        return self._prev

    @data.setter
    def data(self, data: Any) -> None:
        self._data = data

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    @grad.setter
    def grad(self, grad: Any) -> None:
        self._grad = grad

    def __add__(self, other: Value | Any) -> Value:
        other_value: Final[Value] = (other if isinstance(other, Value)
                                     else Value(other))
        result: Final[Value] = Value(self.data + other_value.data,
                                     (self, other_value), '+')

        def _backward() -> None:
            self.grad += result.grad
            other_value.grad += result.grad
        result._backward = _backward

        return result

    def __mul__(self, other: Value | Any) -> Value:
        other_value: Final[Value] = (other if isinstance(other, Value)
                                     else Value(other))
        result: Final[Value] = Value(self.data * other_value.data,
                                     (self, other_value), '*')

        def _backward() -> None:
            self.grad += other_value.data * result.grad
            other_value.grad += self.data * result.grad
        result._backward = _backward

        return result

    def __pow__(self, other: int | float) -> Value:
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        result: Final[Value] = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * result.grad
        result._backward = _backward

        return result

    def relu(self) -> Value:
        result: Final[Value] = Value(
            (0 if self.data < 0 else self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (result.data > 0) * result.grad
        result._backward = _backward

        return result

    def tanh(self) -> Value:
        e_raised_by_2x: Final[float] = math.exp(2 * self.data)
        result: Final[Value] = Value(
            ((e_raised_by_2x + 1) / (e_raised_by_2x - 1)), (self,), "tanh"
        )

        def _backward() -> None:
            self.grad += (1 - result.data ** 2) * result.grad
        result._backward = _backward

        return result

    def __radd__(self, other: Value | Any) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * (-1)

    def __sub__(self, other: Value | Any) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | Any) -> Value:
        return other + (-self)

    def __rmul__(self, other: Value | Any) -> Value:
        return self * other

    def __truediv__(self, other: Value | Any) -> Value:
        return self * (other ** -1)

    def __rtruediv__(self, other: Value | Any) -> Value:
        return other * (self ** -1)

    def exp(self) -> Value:
        result: Final[Value] = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += (math.exp(self.data)) * result.grad
        result._backward = _backward

        return result

    def backward(self) -> None:
        # topological order all of the children in the graph
        topo: Final[list[Value]] = []
        visited: Final[set[Value]] = set()

        def build_topo(v: Value) -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)

        self._grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
