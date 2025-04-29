from abc import ABC, abstractmethod
from typing import Any, Final, Literal
import random

from micrograd.engine import Value


class Module(ABC):

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self) -> list[Value]:
        """
        Abstract method to return the parameters of the module.
        Must be implemented by the subclass.
        """
        pass


class Neuron(Module):

    def __init__(self, in_neurons: int,
                 non_linear: Literal["relu", "tanh", "none"] = "relu") -> None:
        super().__init__()

        self._w: Final[list[Value]] = [
            Value(random.uniform(-1, 1)) for _ in range(in_neurons)
        ]
        self._b: Final[Value] = Value(0)
        self._non_linear: Final[str] = non_linear

    def __call__(self, x: list[Value | Any]) -> Value:
        activation: Final[Value] = sum(
            (wi * xi for wi, xi in zip(self._w, x)), self._b
        )
        if self._non_linear == "relu":
            return activation.relu()
        elif self._non_linear == "tanh":
            return activation.tanh()
        else:
            return activation

    def parameters(self) -> list[Value]:
        return self._w + [self._b]

    def __repr__(self) -> str:
        non_linear: Final[str] = (
            "ReLU" if self._non_linear == "relu" else
            "Tanh" if self._non_linear == "tanh" else
            "Linear"
        )
        return f"{non_linear}Neuron({len(self._w)})"


class Layer(Module):

    def __init__(self, input_cnt: int, output_cnt: int, **kwargs) -> None:
        super().__init__()

        self._neurons: Final[list[Neuron]] = [
            Neuron(input_cnt, **kwargs) for _ in range(output_cnt)
        ]

    def __call__(self, x: list[Value | Any]) -> list[Value] | Value:
        out: Final[list[Value]] = [n(x) for n in self._neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [
            p
            for n in self._neurons
            for p in n.parameters()
        ]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self._neurons)}]"


class MLP(Module):

    def __init__(self, input_cnt: int, output_cnts: list[int]) -> None:
        super().__init__()

        neuron_counts: Final[list[int]] = [input_cnt] + output_cnts
        self._layers: Final[list[Layer]] = [
            Layer(
                input_cnt=neuron_counts[i],
                output_cnt=neuron_counts[i + 1],
                non_linear=(
                    "none" if i == (len(output_cnts) - 1) else "relu"
                )
            )
            for i in range(len(output_cnts))
        ]

    def __call__(self, x: list[Any]) -> list[Value] | Value:
        result: list[Any] | Value = x
        for layer in self._layers:
            assert isinstance(result, list)
            result = layer(result)
        return result

    def __repr__(self) -> str:
        return f"MLP of [{", ".join([str(layer)for layer in self._layers])}]"

    def parameters(self) -> list[Value]:
        return [
            p
            for layer in self._layers
            for p in layer.parameters()
        ]
