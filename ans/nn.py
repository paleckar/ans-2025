from typing import Any, Self

import torch

from ans.autograd import Variable


class Function:
    @classmethod
    def apply(cls, *inputs: Any, **params: Any) -> Variable:
        tensor_args = [i.data if isinstance(i, Variable) else i for i in inputs]
        output_data, cache = cls.forward(*tensor_args, **params)

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, ...]:
            dinputs = cls.backward(dout, cache=cache)
            return tuple(dinputs[i] for i, inp in enumerate(inputs) if isinstance(inp, Variable))

        grad_fn.name = f'{cls.__name__}.backward'
        return Variable(output_data, parents=tuple(i for i in inputs if isinstance(i, Variable)), grad_fn=grad_fn)

    @staticmethod
    def forward(*inputs: torch.Tensor, **params: Any) -> tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}'

    def __repr__(self):
        return str(self)


class BatchNorm1dFunction(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        running_mean: torch.Tensor | None = None,
        running_var: torch.Tensor | None = None,
        momentum: float = 0.1,
        eps: float = 1e-05,
        training: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_features)
            weight: shape (num_features,)
            bias: shape (num_features,)
            running_mean: shape (num_features,)
            running_var: shape (num_features,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_features)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class BatchNorm2dFunction(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        running_mean: torch.Tensor | None = None,
        running_var: torch.Tensor | None = None,
        momentum: float = 0.9,
        eps: float = 1e-05,
        training: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Spatial BatchNorm for convolutional networks

        Args:
            input: shape (num_samples, num_channels, height, width)
            weight: shape (num_channels,)
            bias: shape (num_channels,)
            running_mean: shape (num_channels,)
            running_var: shape (num_channels,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_channels, height, width)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height, width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class DropoutFunction(Function):
    @staticmethod
    def forward(
        input: torch.Tensor,
        p_drop: float = 0.5,
        training: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return (dinput,)


class Conv2dFunction(Function):

    @staticmethod
    def forward(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_channels, height, width)
            weight: shape (num_filters, num_channels, kernel_size[0], kernel_size[1])
            bias: shape (num_filters,)
            stride: convolution step size
            padding: how much should the input be padded on each side by zeroes
            dilation: see torch.nn.functional.conv2d
            groups: see torch.nn.functional.conv2d

        Returns:
            output: shape (num_samples, num_filters, output_height, output_width)
            cache: tuple of intermediate results to use in backward
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_filters, output_height, output_width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input, weight and bias
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class MaxPool2dFunction(Function):

    @staticmethod
    def forward(input: torch.Tensor, kernel_size: int = 2) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_channels, height, width)
            window_size: size of pooling window
        Returns:
            output: shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return (dinput,)


class Module:
    def __init__(self) -> None:
        self.training = True

    def __call__(self, *x: Variable) -> Variable:
        return self.forward(*x)

    def device(self) -> torch.device:
        return next(iter(self.parameters())).data.device

    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).data.dtype

    def forward(self, *x: Variable) -> Variable:
        raise NotImplementedError

    def named_modules(self) -> list[tuple[str, Self]]:
        named_modules = []

        def depth_first_append(obj, prefix=''):
            if isinstance(obj, Module):
                named_modules.append((prefix, obj))
                for name in dir(obj):
                    attr = getattr(obj, name)
                    if isinstance(attr, (list, tuple)):
                        for i, item in enumerate(attr):
                            depth_first_append(item, prefix=f'{prefix}.{i}' if prefix else str(i))
                    else:
                        depth_first_append(attr, prefix=f'{prefix}.{name}' if prefix else name)

        depth_first_append(self)
        return named_modules

    def named_parameters(self) -> list[tuple[str, Variable]]:
        return [
            (f'{name + "." if name else ""}{attr}', getattr(module, attr))
            for name, module in self.named_modules()
            for attr in dir(module)
            if isinstance(getattr(module, attr), Variable)
        ]

    def parameters(self) -> list[Variable]:
        return [p for n, p in self.named_parameters()]

    def num_params(self) -> int:
        return sum(p.data.numel() for p in self.parameters())

    def to(self, dtype: torch.dtype | None = None, device: str | None = None) -> Self:
        def to(obj: Any) -> None:
            if isinstance(obj, torch.Tensor):
                obj.data = obj.to(dtype=dtype, device=device)
            elif isinstance(obj, (tuple, list)):
                for elem in obj:
                    to(elem)
            elif isinstance(obj, dict):
                for val in obj.values():
                    to(val)
            elif isinstance(obj, Variable):
                to(obj.data)
                to(obj.grad)
            elif isinstance(obj, Module):
                for attr in dir(obj):
                    to(getattr(obj, attr))

        to(self)
        return self

    def train(self) -> None:
        for name, layer in self.named_modules():
            layer.training = True

    def eval(self) -> None:
        for name, layer in self.named_modules():
            layer.training = False

    def zero_grad(self) -> None:
        for name, par in self.named_parameters():
            par.grad = None


class Linear(Module):
    def __init__(self, num_in: int, num_out: int) -> None:
        super().__init__()

        ########################################
        # TODO: initialize weight and bias

        self.weight = ...
        self.bias = ...

        # ENDTODO
        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class ReLU(Module):
    def __init__(self, negative_slope: float = 0.0) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Dropout(Module):
    def __init__(self, p_drop: float = 0.5) -> None:
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class BatchNorm1d(Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        ########################################
        # TODO: initialize gamma, beta, running_mean, running_var
        # if affine, then gamma to ones (learnable), otherwise set to None
        # if affine, then beta to zeros (learnable), otherwise set to None
        # running_mean to zeros
        # running_var to ones

        self.weight = ...
        self.bias = ...
        self.running_mean =...
        self.running_var = ...

        # ENDTODO
        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class BatchNorm2d(BatchNorm1d):

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

class Conv2d(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        ########################################
        # TODO: initialize weight and bias
        # if bias is True, then bias should be zeros, otherwise set to None

        self.weight = ...
        self.bias = ...

        # ENDTODO
        ########################################

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class MaxPool2d(Module):

    def __init__(self, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Flatten(Module):

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Optimizer:
    def __init__(self, parameters: list[Variable]) -> None:
        self.parameters = parameters

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None


class SGD(Optimizer):
    def __init__(
        self, parameters: list[Variable], learning_rate: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0
    ) -> None:
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        ########################################
        # TODO: init _velocities to zeros

        self._velocities: dict[Variable, torch.Tensor] = ...

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Variable],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        ########################################
        # TODO: init _num_steps to zero, _m to zeros, _v to zeros

        self._num_steps = ...
        self._m: dict[Variable, torch.Tensor] = ...
        self._v: dict[Variable, torch.Tensor] = ...

        # ENDTODO
        ########################################

    def step(self) -> None:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
