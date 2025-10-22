from numbers import Number
from typing import Callable, Self

import torch


class Variable:
    def __init__(
        self,
        data: torch.Tensor,
        parents: tuple[Self, ...] = (),
        grad_fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.data = data
        self.grad: torch.Tensor | None = None
        self.parents = parents
        self.grad_fn = grad_fn
        self.name = name

    def __repr__(self):
        if hasattr(self.grad_fn, 'func'):
            grad_fn_repr = self.grad_fn.func.__qualname__
        elif self.grad_fn is not None:
            grad_fn_repr = self.grad_fn.__qualname__
        else:
            grad_fn_repr = 'None'
        if self.data.ndim == 0 or (self.data.shape[-1] == self.data.numel() and self.data.numel() < 5):
            return f'{type(self).__name__}({self.data}, grad_fn={grad_fn_repr})'
        else:
            return f'{type(self).__name__}(shape={tuple(self.data.shape)}, grad_fn={grad_fn_repr})'

    def _arg_to_variable(self, arg: Number | torch.Tensor | Self) -> Self:
        if isinstance(arg, Variable):
            return arg
        else:
            if not isinstance(arg, torch.Tensor):
                arg = torch.tensor(arg, dtype=self.data.dtype, device=self.data.device)
            return type(self)(arg)

    def __add__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __radd__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __sub__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __rsub__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __mul__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __rmul__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __truediv__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __rtruediv__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def backprop(self, dout: torch.Tensor | None = None) -> None:
        """
        Runs full backpropagation starting from self. Fills the grad attribute with dself/dpredecessor for all
        predecessors of self.

        Args:
            dout: Incoming gradient on self; if None, then set to tensor of ones with proper shape and dtype
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
