import re
from typing import Any

import gdown
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision

import ans
from tests import ANSTestCase, rand_var
from .test_perceptron_autograd import TestUnaryOp, UnaryOperandType
from .test_neural_library import ANSvsTorchFunctions, ANSvsTorchModules


class TestConv2dFunction(ANSvsTorchFunctions):

    configs = [
        ((11, 4, 13, 16), (8, 5), dict()),
        ((11, 4, 16, 16), (8, 5), dict(stride=2)),
        ((11, 4, 16, 16), (8, 5), dict(stride=2, padding=3)),
        ((11, 4, 16, 16), (8, 5), dict(stride=3, padding=2)),
        ((11, 4, 16, 16), (8, 5), dict(stride=2, padding=4, dilation=2)),
    ]

    def test_implementation(self):
        self.assertCalling(ans.nn.Conv2dFunction.forward, ['conv2d'])
        self.assertNoLoops(ans.nn.Conv2dFunction.forward)
        self.assertCalling(ans.nn.Conv2dFunction.backward, ['conv2d', 'conv_transpose2d'])
        self.assertNotCalling(ans.nn.Conv2dFunction.backward, ['conv2d_input', 'conv2d_weight', 'conv2d_bias'])

    def create_equivalent_functions(self, shape, *args, **kwargs):
        func_ans = lambda x, w, b: ans.nn.Conv2dFunction.apply(x, w, b, **kwargs)
        func_pt = lambda x, w, b: torch.nn.functional.conv2d(x, w, b, **kwargs)
        return func_ans, func_pt

    def random_inputs(self, shape, *args, **kwargs):
        n, c, mx, my = shape
        f, k = args
        g = kwargs.get('groups', 1)
        x = rand_var(n, c, mx, my, name='x', requires_grad=True, dtype=torch.float32)
        w = rand_var(f, c // g, k, k, name='w', requires_grad=True, dtype=torch.float32)
        b = rand_var(f, name='b', requires_grad=True, dtype=torch.float32)
        return x, w, b


class TestConv2dFunctionOptional(TestConv2dFunction):

    configs = [
        ((11, 4, 13, 16), (8, 5), dict()),
        ((11, 4, 16, 16), (8, 5), dict(stride=2)),
        ((11, 4, 16, 16), (8, 5), dict(stride=2, padding=3)),
        ((11, 4, 16, 16), (8, 5), dict(stride=3, padding=2)),
        ((11, 4, 16, 16), (8, 5), dict(stride=2, padding=4, dilation=2)),
        ((11, 4, 16, 16), (8, 5), dict(stride=2, padding=4, dilation=2, groups=2)),
        ((11, 6, 16, 16), (9, 5), dict(stride=2, padding=4, dilation=2, groups=3)),
    ]


class TestConv2dModule(ANSvsTorchModules):

    configs = [
        ((10, 45, 6, 9), (45, 277, 3), dict()),
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        c, f, k = args
        module_ans = ans.nn.Conv2d(c, f, k, **kwargs)
        module_pt = torch.nn.Conv2d(c, f, k, **kwargs)
        return module_ans, module_pt

    def check_init(self, module_ans, module_pt):
        # Check types
        self.assertIsInstance(module_ans.weight, ans.autograd.Variable)
        self.assertIsInstance(module_ans.bias, ans.autograd.Variable)
        self.assertTupleEqual(module_ans.weight.data.shape, module_pt.weight.shape)
        self.assertTupleEqual(module_ans.bias.data.shape, module_pt.bias.shape)

        # Check init values distribution
        r_min = min(module_ans.weight.data.min().item(), module_pt.weight.min().item())
        r_max = min(module_ans.weight.data.max().item(), module_pt.weight.max().item())
        h_ans, _ = torch.histogram(module_ans.weight.data.view(-1), 10, range=(r_min, r_max), density=True)
        h_pt, _ = torch.histogram(module_pt.weight.data.view(-1), 10, range=(r_min, r_max), density=True)
        self.assertTensorsClose(h_ans, h_pt, rtol=0.05, atol=0.0)


class TestMaxPool2dFunction(ANSvsTorchFunctions):

    configs = [
        ((6, 4, 5, 8), (2,), dict()),
        ((6, 4, 5, 8), (3,), dict()),
        ((6, 4, 5, 8), (5,), dict()),
    ]

    def test_implementation(self):
        self.assertNotCalling(ans.nn.MaxPool2dFunction.forward, ['max_pool2d'])
        self.assertNoLoops(ans.nn.MaxPool2dFunction.forward)
        self.assertNoLoops(ans.nn.MaxPool2dFunction.backward)

    def create_equivalent_functions(self, shape, *args, **kwargs):
        (k,) = args
        func_ans = lambda x: ans.nn.MaxPool2dFunction.apply(x, k, **kwargs)
        func_pt = lambda x: torch.nn.functional.max_pool2d(x, k, **kwargs)
        return func_ans, func_pt

    def random_inputs(self, shape, *args, **kwargs):
        x = rand_var(*shape, name='x', requires_grad=True, dtype=torch.float32)
        return (x,)


class TestMaxPool2dModule(ANSvsTorchModules):

    configs = [
        ((6, 4, 5, 8), (2,), dict()),
        ((6, 4, 5, 8), (3,), dict()),
        ((6, 4, 5, 8), (5,), dict()),
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        module_ans = ans.nn.MaxPool2d(*args, **kwargs)
        module_pt = torch.nn.MaxPool2d(*args, **kwargs)
        return module_ans, module_pt


class TestReshapeVariable(TestUnaryOp):

    configs = [
        ((2,), dict(shape=(1, 2))),
        ((2, 3), dict(shape=(6,))),
        ((2, 3, 4), dict(shape=(4, 2, 3))),
        ((2, 3, 4), dict(shape=(4 * 2, 3))),
        ((2, 3, 4), dict(shape=(4, -1))),
        ((2, 3, 4), dict(shape=(-1, 1))),
        ((2, 3, 4), dict(shape=(-1,))),
    ]
    dds = [  # dtypes and devices to test
        (torch.float32, 'cpu'),
        (torch.float64, 'cpu'),
        (torch.float32, 'meta'),
    ]

    @staticmethod
    def forward(x: UnaryOperandType, **kwargs: Any) -> UnaryOperandType:
        return x.reshape(kwargs['shape'])


class TestFlattenModule(ANSvsTorchModules):

    configs = [
        ((6, 4), tuple(), dict()),
        ((6, 4, 5), tuple(), dict()),
        ((6, 4, 5, 8), tuple(), dict()),
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        module_ans = ans.nn.Flatten(*args, **kwargs)
        module_pt = torch.nn.Flatten(*args, **kwargs)
        return module_ans, module_pt


class TestBatchNorm2dFunction(ANSvsTorchFunctions):

    configs = [
        ((11, 4, 13, 16), tuple(), dict(training=True)),
        ((11, 4, 13, 16), tuple(), dict(training=False)),
    ]

    def test_implementation(self):
        self.assertNotCalling(ans.nn.BatchNorm2dFunction.forward, ['batch_norm'])

    def create_equivalent_functions(self, shape, *args, **kwargs):
        n, c, h, w = shape
        run_mean, run_var = torch.randn(c), torch.rand(c)
        momentum = torch.rand(1).item()
        func_ans = lambda x, w, b: ans.nn.BatchNorm2dFunction.apply(
            x, w, b, run_mean, run_var, momentum=momentum, training=kwargs['training']
        )
        func_pt = lambda x, w, b: torch.nn.functional.batch_norm(
            x, run_mean, run_var, weight=w, bias=b, momentum=momentum, training=kwargs['training']
        )
        return func_ans, func_pt

    def random_inputs(self, shape, *args, **kwargs):
        n, c, h, w = shape
        x_var = rand_var(n, c, h, w, name='x', requires_grad=True, dtype=torch.float32, rng_fn=torch.rand)
        w_var = rand_var(c, name='w', requires_grad=True, dtype=torch.float32)
        b_var = rand_var(c, name='b', requires_grad=True, dtype=torch.float32)
        return x_var, w_var, b_var


class TestBatchNorm2dModule(ANSvsTorchModules):

    configs = [
        ((11, 4, 13, 16), tuple(), dict(affine=False)),
        ((11, 4, 13, 16), tuple(), dict(affine=True)),
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        n, c, h, w = shape
        momentum = torch.rand(1).item()
        eps = 0.01 * torch.rand(1).item()
        module_ans = ans.nn.BatchNorm2d(c, momentum=momentum, affine=kwargs['affine'], eps=eps)
        module_pt = torch.nn.BatchNorm2d(c, momentum=momentum, affine=kwargs['affine'], eps=eps)
        return module_ans, module_pt

    def check_init(self, module_ans, module_pt):
        if module_pt.affine:
            self.assertTensorsClose(module_ans.weight.data, module_pt.weight)
            self.assertTensorsClose(module_ans.bias.data, module_pt.bias)


class TestBackbone(ANSTestCase):

    configs = [
        ((11, 3, 32, 32), (7,), dict()),
        ((11, 4, 32, 32), (7,), dict()),
        ((11, 4, 32, 32), (8,), dict()),
        ((11, 4, 33, 48), (8,), dict()),
    ]

    def monkey_patch_convs(self) -> None:
        self.conv_counter = 0
        self.orig_conv_forward = ans.nn.Conv2dFunction.forward

        def counted_conv_forward(*args, **kwargs):
            self.conv_counter += 1
            return self.orig_conv_forward(*args, **kwargs)

        ans.nn.Conv2dFunction.forward = counted_conv_forward

    def monkey_unpatch_convs(self) -> None:
        ans.nn.Conv2dFunction.forward = self.orig_conv_forward

    def test_backbone(self):
        backbone_cls = self.params['backbone_cls']

        for (n, c, h, w), (k,), kwargs in self.configs:
            backbone = backbone_cls(c, k)
            self.assertIsInstance(backbone, ans.nn.Module)
            self.assertTrue(any(isinstance(m, ans.nn.Conv2d) for _, m in backbone.named_modules()))

            x_var = rand_var(n, c, h, w, name='x', dtype=torch.float32)
            try:
                self.monkey_patch_convs()
                z_var = backbone(x_var)
            finally:
                self.monkey_unpatch_convs()
            self.assertGreater(self.conv_counter, 0, msg='Model must use convolution')
            self.assertEqual(z_var.data.ndim, 2, msg=f'Outputs should be {n} x {k} logits')
            self.assertEqual(z_var.data.size(0), x_var.data.size(0), msg=f'Outputs should be {n} x {k} logits')
            self.assertEqual(z_var.data.size(1), k, msg=f'Outputs should be {n} x {k} logits')


class TestDataPreprocessor(ANSTestCase):

    def check_dataset(self, orig_ds, prep_ds, train=False):
        x_orig = orig_ds.data
        x_prep, y_prep = prep_ds.tensors
        self.assertEqual(x_prep.ndim, 4, msg='Preprocessed dataset must be 4D tensor (N, C, H, W)')
        if not train:
            self.assertEqual(len(x_prep), len(x_orig), msg='Augmentation not allowed in validation set')
        self.assertEqual(x_prep[0].numel(), x_orig[0].size, msg='Resizing and/or feature extraction not allowed')
        self.assertEqual(y_prep.size(0), x_prep.size(0))
        self.assertEqual(y_prep.ndim, 1, msg='Targets must be a vector of integers')

    def test_preprocess(self):
        train_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=True,
        )
        preprocessor = self.params['preprocessor_cls']()
        preprocessor.fit(train_dataset)
        self.check_dataset(train_dataset, preprocessor.transform(train_dataset, train=True), train=True)
        val_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
        )
        self.check_dataset(val_dataset, preprocessor.transform(val_dataset, train=False), train=False)


class TestValAccuracy75(ANSTestCase):

    min_val_acc = 0.75

    def check_val_acc(self, model, device='cuda'):
        model.to(device=device)

        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        val_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)

        preprocessor = self.params['preprocessor_cls']()
        preprocessor.fit(train_dataset)
        train_dataset = preprocessor.transform(train_dataset, train=True)
        val_dataset = preprocessor.transform(val_dataset, train=False)

        train_loader = ans.data.BatchLoader(train_dataset, batch_size=100, shuffle=True, device=device)
        val_loader = ans.data.BatchLoader(val_dataset, batch_size=100, shuffle=False, device=device)

        train_loss, train_acc = ans.classification.validate(model, train_loader)
        val_loss, val_acc = ans.classification.validate(model, val_loader)
        self.assertGreaterEqual(val_acc, self.min_val_acc)
        self.assertLess(val_acc, train_acc + 0.02)
        self.assertGreater(val_loss, train_loss - 0.1)

    def test_val_acc(self):
        try:
            model = ans.classification.AutogradClassifier.load('../output/conv_classifier.pt')
        except FileNotFoundError:
            with open('../output/conv_classifier.gdrive') as f:
                model_link = f.read().strip()
            file_id = re.search(r'[a-zA-Z0-9_-]{33,}', model_link).group()
            model_link = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(model_link, '../output/conv_classifier.pt', quiet=False)
            model = ans.classification.AutogradClassifier.load('../output/conv_classifier.pt')
        try:
            self.check_val_acc(model, device='cuda')
        except RuntimeError:
            print("device='cuda' not available/implemented")
            self.check_val_acc(model, device='cpu')


class TestValAccuracy85(TestValAccuracy75):

    min_val_acc = 0.85


class TestValAccuracy90(TestValAccuracy75):

    min_val_acc = 0.90
