from numbers import Number
from typing import Any, TypeAlias

import torch

from ans.autograd import Variable
from tests import ANSTestCase, rand_var


BinaryOperandType: TypeAlias = Number | torch.Tensor | Variable


class TestBinaryOpTensors(ANSTestCase):
    shape_pairs = [
        (tuple(), tuple()),
        ((2,), (2,)),
        ((2, 3), (2, 3)),
        ((2, 3), tuple()),
        (tuple(), (2, 3)),
        ((3,), (2, 3)),
        ((2, 3), (2, 1)),
    ]
    dds = [  # dtypes and devices to test
        (torch.float32, 'cpu'),
        (torch.float64, 'cpu'),
        (torch.float32, 'meta'),
    ]
    input_types = [
        (Variable, Variable),
        (Variable, Number),
        (Variable, torch.Tensor),
        (Number, Variable),
        (torch.Tensor, Variable),
    ]

    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        raise NotImplementedError

    def test_operation(self):
        def _totype(v: Variable, t: BinaryOperandType) -> BinaryOperandType:
            if t == Number:
                return v.data.item()
            elif t == torch.Tensor:
                return v.data
            else:
                return v

        for x_shape, y_shape in self.shape_pairs:
            for dtype, device in self.dds:
                for x_type, y_type in self.input_types:
                    msg = (
                        f'\n*** TEST PARAMS CONFIGURATION THAT FAILED ***\n'
                        f'x_shape={x_shape}, y_shape={y_shape}, dtype={dtype}, device={device}, x_type={x_type}, y_type={y_type}'
                    )
                    x_var = rand_var(*x_shape, requires_grad=True, dtype=dtype, device=device)
                    y_var = rand_var(*y_shape, requires_grad=True, dtype=dtype, device=device)

                    try:
                        x = _totype(x_var, x_type)
                        y = _totype(y_var, y_type)
                    except RuntimeError:  # Number + vector shape combination will fail on .item()
                        continue

                    # forward pass
                    try:
                        z = self.forward(x, y)  # with Variables
                    except Exception:
                        print(msg)  # for debug purposes
                        raise
                    expected_z = self.forward(getattr(x, 'data', x), getattr(y, 'data', y))  # with Tensors
                    self.assertTensorsClose(z.data, expected_z, msg=msg)
                    self.assertIsInstance(z, Variable)
                    self.assertIsNotNone(z.grad_fn)
                    self.assertIsNone(z.grad)
                    self.assertIsInstance(z.parents, tuple)
                    self.assertEqual(len(z.parents), 2)
                    if isinstance(x, Variable):
                        self.assertIs(z.parents[0], x)
                        self.assertIsNone(x.grad)
                    if isinstance(y, Variable):
                        self.assertIs(z.parents[1], y)
                        self.assertIsNone(y.grad)

                    # backward pass
                    dz = torch.randn_like(z.data)
                    expected_z.backward(gradient=dz)
                    try:
                        dx, dy = z.grad_fn(dz)
                    except Exception:
                        print(msg)  # for debug pruposes
                        raise
                    if isinstance(x, Variable):
                        self.assertTensorsClose(dx, x_var.data.grad, msg=msg)
                    if isinstance(y, Variable):
                        self.assertTensorsClose(dy, y_var.data.grad, msg=msg)


class TestBinaryOpScalars(TestBinaryOpTensors):
    shape_pairs = [
        (tuple(), tuple()),
    ]
    dds = [
        (torch.float32, 'cpu'),
    ]
    input_types = [
        (Variable, Variable),
    ]


class TestAddScalars(TestBinaryOpScalars):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x + y


class TestSubScalars(TestBinaryOpScalars):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x - y


class TestMulScalars(TestBinaryOpScalars):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x * y


class TestDivScalars(TestBinaryOpScalars):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x / y


class TestAddTensors(TestBinaryOpTensors):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x + y


class TestSubTensors(TestBinaryOpTensors):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x - y


class TestMulTensors(TestBinaryOpTensors):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x * y


class TestDivTensors(TestBinaryOpTensors):
    @staticmethod
    def forward(x: BinaryOperandType, y: BinaryOperandType) -> BinaryOperandType:
        return x / y


class TestBackprop(ANSTestCase):
    shapes = (tuple(), (2,), (2, 3))
    dds = [  # dtypes and devices to test
        (torch.float32, 'cpu'),
        (torch.float64, 'cpu'),
        (torch.float32, 'meta'),
    ]

    def __init__(self, methodName: str = '', **params: Any):
        super().__init__(methodName, **params)
        self.examples = [
            example_1,
            example_2,
            example_3,
        ]

    def test_backprop(self):
        def counted_grad_fn(var: Variable):
            grad_fn = var.grad_fn

            def wrapper(*args, **kwargs):
                call_counts[var] += 1
                return grad_fn(*args, **kwargs)

            return wrapper

        for shape in self.shapes:
            for dtype, device in self.dds:
                for example_fn in self.examples:
                    for shape in self.shapes:
                        variables = example_fn(shape, dtype=dtype, device=device)
                        call_counts = {var: 0 for var in variables}
                        for var in variables:
                            if var.parents:
                                var.grad_fn = counted_grad_fn(var)  # count how many times grad_fn has been called
                        dout = torch.randn_like(variables[-1].data)
                        variables[-1].backprop(dout=dout)  # ans backprop
                        for var in variables:
                            var.data.retain_grad()  # to check intermediate gradients even though they don't really matter
                        variables[-1].data.backward(gradient=dout)  # pytorch backprop as reference
                        for var in variables:
                            self.assertTensorsClose(var.grad, var.data.grad)
                            expected_call_count = (
                                1 if var.parents else 0
                            )  # each node's chainrule must be called at most once
                            self.assertEqual(call_counts[var], expected_call_count)


def example_1(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device(type='cpu'),
) -> tuple[Variable, ...]:
    u = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    v = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    w = u * v
    return u, v, w


def example_2(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device(type='cpu'),
) -> tuple[Variable, ...]:
    x1 = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    a = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    x2 = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    x2_ = a * x2
    y = x1 + x2_
    z = y * y
    return x1, a, x2, x2_, y, z


def example_3(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device(type='cpu'),
) -> tuple[Variable, ...]:
    x = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    o = rand_var(*shape, requires_grad=True, dtype=dtype, device=device)
    s = x * x
    p = s + o
    m = s - o
    q = p * m
    return x, o, s, p, m, q
