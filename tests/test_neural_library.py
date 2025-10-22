from typing import Any
import numpy as np
import PIL
import torch
import torch.utils
import torch.utils.data
import torchvision

import ans
from tests import ANSTestCase, rand_var
from .test_perceptron_autograd import TestUnaryOp, UnaryOperandType
from .test_linear_classification import TestBatchLoader


class ANSvsTorchParamsTest(ANSTestCase):

    def clone_params(self, module_ans, module_pt, transpose=False):
        params_ans = sorted(module_ans.named_parameters())
        params_pt = sorted(module_pt.named_parameters())
        self.assertEqual(len(params_ans), len(params_pt))
        for (_, apar), (_, tpar) in zip(params_ans, params_pt):
            tpar.data = apar.data.clone() if not transpose else apar.data.clone().t()


class ANSvsTorchModules(ANSvsTorchParamsTest):

    configs = []
    trainings = [True, False]
    
    def assertParamsClose(self, module_ans, module_pt):
        for (_, apar), (_, tpar) in zip(sorted(module_ans.named_parameters()), sorted(module_pt.named_parameters())):
            self.assertTensorsClose(apar.grad, tpar.grad.t())
    
    def create_equivalent_modules(self, shape, *args, **kwargs):
        pass

    def check_init(self, module_ans, module_pt):
        pass

    def check_output(self, res_ans, res_pt):
        self.assertTensorsClose(res_ans, res_pt)
    
    def check_grads(self, module_ans, module_pt, x_var):
        for (an, apar), (tn, tpar) in zip(sorted(module_ans.named_parameters()), sorted(module_pt.named_parameters())):
            self.assertTensorsClose(apar.grad, tpar.grad)
        self.assertTensorsClose(x_var.grad, x_var.data.grad)

    def random_input(self, *shape):
        return rand_var(*shape, requires_grad=True, dtype=torch.float32)
    
    def check_forward_pass(self, module_ans, module_pt, x_var):
        z_var = module_ans(x_var)
        z = module_pt(x_var.data)
        self.assertIsInstance(z_var, ans.autograd.Variable)
        self.check_output(z_var.data, z)
        return z_var, z
    
    def check_backward_pass(self, module_ans, module_pt, x_var, z_var, z):
        dz = torch.randn_like(z)
        z_var.backprop(dout=dz)
        z.backward(gradient=dz)
        self.check_grads(module_ans, module_pt, x_var)
    
    def test_module(self):
        for shape, args, kwargs in self.configs:
            msg = (f"\n*** TEST PARAMS CONFIGURATION THAT FAILED ***\n"
                       f"shape={shape}, args={args}, kwargs={kwargs}")
            try:
                module_ans, module_pt = self.create_equivalent_modules(shape, *args, **kwargs)
                self.check_init(module_ans, module_pt)
                self.clone_params(module_ans, module_pt)
                for training in self.trainings:
                    if training:
                        module_ans.train()
                        module_pt.train()
                    else:
                        module_ans.eval()
                        module_pt.eval()
                    x_var = self.random_input(*shape)
                    z_var, z = self.check_forward_pass(module_ans, module_pt, x_var)
                    self.check_backward_pass(module_ans, module_pt, x_var, z_var, z)
            except Exception:
                print(msg)
                raise


class TestLinearModule(ANSvsTorchModules):

    configs = [
        ((13, 400), (400, 300), dict())
    ]

    def clone_params(self, module_ans, module_pt, transpose=False):
        return super().clone_params(module_ans, module_pt, transpose=True)
    
    def create_equivalent_modules(self, shape, *args, **kwargs):
        module_ans = ans.nn.Linear(*args, **kwargs)
        module_pt = torch.nn.Linear(*args, **kwargs)
        return module_ans, module_pt

    def check_init(self, module_ans, module_pt):
        # Check types
        self.assertIsInstance(module_ans.weight, ans.autograd.Variable)
        self.assertIsInstance(module_ans.bias, ans.autograd.Variable)
        self.assertTupleEqual(module_ans.weight.data.shape, module_pt.weight.t().shape)
        self.assertTupleEqual(module_ans.bias.data.shape, module_pt.bias.shape)

        # Check init values distribution
        r_min = min(module_ans.weight.data.min().item(), module_pt.weight.min().item())
        r_max = min(module_ans.weight.data.max().item(), module_pt.weight.max().item())
        h_ans, _ = torch.histogram(module_ans.weight.data.view(-1), 10, range=(r_min, r_max), density=True)
        h_pt, _ = torch.histogram(module_pt.weight.data.view(-1), 10, range=(r_min, r_max), density=True)
        self.assertTensorsClose(h_ans, h_pt, rtol=0.05, atol=0.0)
    
    def check_grads(self, module_ans, module_pt, x_var):
        for (an, apar), (tn, tpar) in zip(sorted(module_ans.named_parameters()), sorted(module_pt.named_parameters())):
            self.assertTensorsClose(apar.grad, tpar.grad.t())  # torch.nn.Linear does x * w.t() + b
        self.assertTensorsClose(x_var.grad, x_var.data.grad)


class TestSigmoidModule(ANSvsTorchModules):

    configs = [
        ((13, 400), tuple(), dict()),
        ((13, 400, 300), tuple(), dict())
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        return ans.nn.Sigmoid(), torch.nn.Sigmoid()


class TestReLUVariable(TestUnaryOp):

    dds = [  # dtypes and devices to test
        (torch.float32, 'cpu'),
        (torch.float64, 'cpu'),
        # (torch.float32, 'meta'),  # causes RuntimeError
    ]

    @staticmethod
    def forward(x: UnaryOperandType, **kwargs: Any) -> UnaryOperandType:
        return x.relu()


class TestReLUModule(ANSvsTorchModules):

    def create_equivalent_modules(self, shape, *args, **kwargs):
        return ans.nn.ReLU(), torch.nn.ReLU()


class TestSequentialModule(TestLinearModule):

    configs = [
        ((13, 6), (5, 4, 3), dict())
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        n, d = shape
        h1, h2, h3 = args
        module_ans = ans.nn.Sequential(
            ans.nn.Linear(d, h1),
            ans.nn.Sigmoid(),
            ans.nn.Linear(h1, h2),
            ans.nn.ReLU(),
            ans.nn.Linear(h2, h3)
        )
        module_pt = torch.nn.Sequential(
            torch.nn.Linear(d, h1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, h3)
        )
        return module_ans, module_pt
    
    def check_init(self, module_ans, module_pt):
        pass


class TestSGD(ANSvsTorchParamsTest):

    def setUp(self) -> None:
        self.model_ans = ans.nn.Sequential(
            ans.nn.Linear(6, 5),
            ans.nn.Sigmoid(),
            ans.nn.Linear(5, 4),
            ans.nn.Sigmoid(),
            ans.nn.Linear(4, 3)
        )
        self.model_pt = torch.nn.Sequential(
            torch.nn.Linear(6, 5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(5, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 3)
        )
        self.clone_params(self.model_ans, self.model_pt, transpose=True)

    def test_init(self) -> None:
        model = ans.nn.Linear(4, 4).to(dtype=torch.float16)
        optimizer = ans.nn.SGD(model.parameters())
        v = next(iter(optimizer._velocities.values()))
        self.assertIsInstance(v, torch.Tensor)
        self.assertEqual(v.dtype, torch.float16)
        self.assertTrue(torch.all(v == 0))

    def _test_config(self, learning_rate: float, weight_decay: float, momentum: float) -> None:
        ans_optimizer = ans.nn.SGD(
            self.model_ans.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )
        torch_optimizer = torch.optim.SGD(
            self.model_pt.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        for i in range(3):
            for (_, apar), (_, tpar) in zip(sorted(self.model_ans.named_parameters()), sorted(self.model_pt.named_parameters())):
                apar.grad = 0.1 * torch.randn_like(apar.data)
                tpar.grad = apar.grad.clone().t()
            ans_optimizer.step()
            torch_optimizer.step()
        for (_, apar), (_, tpar) in zip(sorted(self.model_ans.named_parameters()), sorted(self.model_pt.named_parameters())):
            self.assertTensorsClose(apar.data, tpar.t())

    def test_sgd(self) -> None:
        self._test_config(torch.rand(1).item(), 0., 0.)

    def test_weight_decay(self) -> None:
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), 0.)

    def test_momentum(self) -> None:
        self._test_config(torch.rand(1).item(), 0., torch.rand(1).item())

    def test_weight_decay_momentum(self) -> None:
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())


class TestAdam(ANSvsTorchParamsTest):

    def setUp(self) -> None:
        self.model_ans = ans.nn.Sequential(
            ans.nn.Linear(6, 5),
            ans.nn.Sigmoid(),
            ans.nn.Linear(5, 4),
            ans.nn.Sigmoid(),
            ans.nn.Linear(4, 3)
        )
        self.model_pt = torch.nn.Sequential(
            torch.nn.Linear(6, 5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(5, 4),
            torch.nn.Sigmoid(),
            torch.nn.Linear(4, 3)
        )
        self.clone_params(self.model_ans, self.model_pt, transpose=True)

    def test_init(self) -> None:
        model = ans.nn.Linear(4, 4).to(dtype=torch.float16)
        optimizer = ans.nn.Adam(model.parameters())
        for a in ('_m', '_v'):
            val = next(iter(getattr(optimizer, a).values()))
            self.assertIsInstance(val, torch.Tensor)
            self.assertEqual(val.dtype, torch.float16)
            self.assertTrue(torch.all(val == 0))

    def _test_config(self, learning_rate: float, weight_decay: float, beta1: float, beta2: float):
        ans_optimizer = ans.nn.Adam(
            self.model_ans.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            eps=torch.rand(1).item()
        )
        torch_optimizer = torch.optim.Adam(
            self.model_pt.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=ans_optimizer.eps
        )
        for i in range(3):
            for (_, apar), (_, tpar) in zip(sorted(self.model_ans.named_parameters()), sorted(self.model_pt.named_parameters())):
                apar.grad = 0.1 * torch.randn_like(apar.data)
                tpar.grad = apar.grad.clone().t()
            ans_optimizer.step()
            torch_optimizer.step()
        for (_, apar), (_, tpar) in zip(sorted(self.model_ans.named_parameters()), sorted(self.model_pt.named_parameters())):
            self.assertTensorsClose(apar.data, tpar.t())

    def test_adam(self):
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())
        self._test_config(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())


class ANSvsTorchFunctions(ANSTestCase):

    configs = []

    def create_equivalent_functions(self, shape, *args, **kwargs):
        raise NotImplementedError

    def random_inputs(self, shape, *args, **kwargs):
        raise NotImplementedError
    
    def check_output(self, res_ans, res_pt):
        self.assertTensorsClose(res_ans, res_pt)
    
    def check_grads(self, inp_vars):
        for in_var in inp_vars:
            msg = f"d{in_var.name} " if in_var.name is not None else ''
            self.assertTensorsClose(in_var.grad, in_var.data.grad, msg=msg)
    
    def check_forward_pass(self, func_ans, func_pt, inp_vars):
        z_var = func_ans(*inp_vars)
        z = func_pt(*[iv.data for iv in inp_vars])
        self.assertIsInstance(z_var, ans.autograd.Variable)
        self.check_output(z_var.data, z)
        return z_var, z
    
    def check_backward_pass(self, inp_vars, z_var, z):
        dz = torch.randn_like(z)
        z_var.backprop(dout=dz)
        z.backward(gradient=dz)
        self.check_grads(inp_vars)
    
    def test_function(self):
        for shape, args, kwargs in self.configs:
            msg = (f"\n*** TEST PARAMS CONFIGURATION THAT FAILED ***\n"
                       f"shape={shape}, args={args}, kwargs={kwargs}")
            try:
                func_ans, func_pt = self.create_equivalent_functions(shape, *args, **kwargs)
                inp_vars = self.random_inputs(shape, *args, **kwargs)
                z_var, z = self.check_forward_pass(func_ans, func_pt, inp_vars)
                self.check_backward_pass(inp_vars, z_var, z)
            except Exception:
                print(msg)
                raise


class TestDropoutFunction(ANSvsTorchFunctions):

    configs = [
        ((100, 400), tuple(), dict(p=0.2, training=True)),
        ((100, 400), tuple(), dict(p=0.5, training=True)),
        ((100, 400), tuple(), dict(p=0.8, training=True)),
        ((100, 400), tuple(), dict(p=0.2, training=False)),
        ((100, 400), tuple(), dict(p=0.5, training=False)),
        ((100, 400), tuple(), dict(p=0.8, training=False)),
    ]

    def test_implementation(self):
        self.assertNotCalling(ans.nn.DropoutFunction.forward, ['dropout', 'dropout_'])

    def create_equivalent_functions(self, shape, *args, **kwargs):
        func_ans = lambda x: ans.nn.DropoutFunction.apply(x, p_drop=kwargs['p'], training=kwargs['training'])
        func_pt = lambda x: torch.nn.functional.dropout(x, p=kwargs['p'], training=kwargs['training'])
        return func_ans, func_pt
    
    def random_inputs(self, shape, *args, **kwargs):
        x_var = rand_var(*shape, mean=1.0, requires_grad=True, rng_fn=torch.rand)
        return x_var,

    def check_output(self, res_ans, res_pt):
        self.assertTensorsClose((res_ans == 0.0).float().mean(), (res_pt == 0.0).float().mean(), rtol=0.05, atol=0.0)

    def check_grads(self, inp_vars):
        for in_var in inp_vars: 
            self.assertTensorsClose((in_var.grad == 0.0).float().mean(), (in_var.data.grad == 0.0).float().mean(), rtol=0.05, atol=0.0)
            mask = (in_var.grad != 0.0) & (in_var.data.grad != 0.0)
            self.assertTensorsClose(in_var.grad[mask], in_var.data.grad[mask])


class TestDropoutModule(ANSvsTorchModules):

    configs = [
        ((100, 400), tuple(), dict(p=0.2)),
        ((100, 400), tuple(), dict(p=0.5)),
        ((100, 400), tuple(), dict(p=0.8)),
        ((100, 400, 30), tuple(), dict(p=0.8)),
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        return ans.nn.Dropout(p_drop=kwargs['p']), torch.nn.Dropout(p=kwargs['p'])

    def check_output(self, res_ans, res_pt):
        self.assertTensorsClose((res_ans == 0.0).float().mean(), (res_pt == 0.0).float().mean(), rtol=0.05, atol=0.0)
    
    def check_grads(self, module_ans, module_pt, x_var):
        for (_, apar), (_, tpar) in zip(sorted(module_ans.named_parameters()), sorted(module_pt.named_parameters())):
            self.assertTensorsClose((apar.grad == 0.0).float().mean(), (apar.data.grad == 0.0).float().mean(), rtol=0.05, atol=0.0)
        self.assertTensorsClose((x_var.grad == 0.0).float().mean(), (x_var.data.grad == 0.0).float().mean(), rtol=0.05, atol=0.0)


class TestBatchNorm1dFunction(ANSvsTorchFunctions):

    configs = [
        ((100, 40), tuple(), dict(training=True)),
        ((100, 40), tuple(), dict(training=False)),
    ]

    def test_implementation(self):
        self.assertNotCalling(ans.nn.DropoutFunction.forward, ['batch_norm'])

    def create_equivalent_functions(self, shape, *args, **kwargs):
        n, d = shape
        run_mean, run_var = torch.randn(d), torch.rand(d)
        momentum = torch.rand(1).item()
        func_ans = lambda x, w, b: ans.nn.BatchNorm1dFunction.apply(x, w, b, run_mean, run_var, momentum=momentum, training=kwargs['training'])
        func_pt = lambda x, w, b: torch.nn.functional.batch_norm(x, run_mean, run_var, weight=w, bias=b, momentum=momentum, training=kwargs['training'])
        return func_ans, func_pt
    
    def random_inputs(self, shape, *args, **kwargs):
        n, d = shape
        x_var = rand_var(n, d, requires_grad=True, dtype=torch.float32, rng_fn=torch.rand)
        w_var = rand_var(d, requires_grad=True, dtype=torch.float32)
        b_var = rand_var(d, requires_grad=True, dtype=torch.float32)
        return x_var, w_var, b_var


class TestBatchNorm1dModule(ANSvsTorchModules):

    configs = [
        ((100, 40), tuple(), dict(affine=False)),
        ((100, 40), tuple(), dict(affine=True)),
    ]

    def create_equivalent_modules(self, shape, *args, **kwargs):
        n, d = shape
        momentum = torch.rand(1).item()
        eps = 0.01 * torch.rand(1).item()
        module_ans = ans.nn.BatchNorm1d(d, momentum=momentum, affine=kwargs['affine'], eps=eps)
        module_pt = torch.nn.BatchNorm1d(d, momentum=momentum, affine=kwargs['affine'], eps=eps)
        return module_ans, module_pt
    
    def check_init(self, module_ans, module_pt):
        if module_pt.affine:
            self.assertTensorsClose(module_ans.weight.data, module_pt.weight)
            self.assertTensorsClose(module_ans.bias.data, module_pt.bias)
    
    def clone_params(self, module_ans, module_pt):
        if module_pt.affine:
            module_ans.weight.data = torch.randn_like(module_pt.weight)
            module_ans.bias.data = torch.randn_like(module_pt.bias)
        return super().clone_params(module_ans, module_pt)


class TestAutogradClassifier(ANSTestCase):

    def setUp(self) -> None:
        self.inputs = ans.autograd.Variable(torch.tensor([
            [0.6803, 0.5091, 0.9956, 0.3223],
            [0.0412, 0.2676, 0.8497, 0.8004],
            [0.8800, 0.1041, 0.3104, 0.8930],
            [0.3836, 0.4640, 0.4453, 0.0873],
            [0.0481, 0.6611, 0.3160, 0.0896],
            [0.0683, 0.7224, 0.0070, 0.7719],
            [0.3652, 0.6974, 0.0702, 0.0464]
        ]))
        self.targets = torch.tensor([1, 3, 4, 4, 4, 1, 0])
        self.linear_1 = ans.nn.Linear(6, 4)
        self.linear_1.weight.data = torch.tensor([
            [ 0.2406,  0.3387, -0.2115,  0.0232, -0.2838, -0.1691],
            [ 0.1583, -0.4303,  0.2484, -0.3102,  0.0281, -0.4459],
            [-0.1948, -0.0103, -0.2441, -0.1064, -0.0958,  0.1656],
            [ 0.4526,  0.0814,  0.1609,  0.2857,  0.2971, -0.1224]
        ])
        self.linear_1.bias.data = torch.tensor([0.6371, 0.2196, 0.8141, 0.8793, 0.7719, 0.6085])
        self.linear_2 = ans.nn.Linear(4, 3)
        self.linear_2.weight.data = torch.tensor([
            [ 0.1480, -0.3501, -0.2098],
            [-0.3525, -0.3964, -0.1268],
            [ 0.3726,  0.2080, -0.4046],
            [-0.0085,  0.2890,  0.0756],
            [-0.2304,  0.1066, -0.3855],
            [-0.0759,  0.0859, -0.0955]
        ])
        self.linear_2.bias.data = torch.tensor([0.4585, 0.5676, 0.7828])
        self.linear_3 = ans.nn.Linear(3, 5)
        self.linear_3.weight.data = torch.tensor([
            [ 0.0834,  0.2429,  0.1635,  0.4697, -0.4852],
            [ 0.5094,  0.1167, -0.1539,  0.5173,  0.1496],
            [ 0.3888, -0.5434, -0.4701,  0.3005,  0.0418]
        ])
        self.linear_3.bias.data = torch.tensor([1.1117, 0.8112, 0.2674, 0.8977, 0.1446])
        self.backbone = ans.nn.Sequential(
            self.linear_1,
            ans.nn.Sigmoid(),
            self.linear_2,
            ans.nn.ReLU(),
            self.linear_3
        )
        self.optimizer = ans.nn.SGD(self.backbone.parameters(), learning_rate=0.8, momentum=0.0, weight_decay=0.0)
        self.model = ans.classification.AutogradClassifier(self.backbone, self.optimizer)

    def test_implementation(self):
        self.assertCalling(
            ans.classification.AutogradClassifier.train_step,
            ['backbone', 'softmax_cross_entropy', 'zero_grad', 'backprop', 'step']
        )
        self.assertCalling(
            ans.classification.AutogradClassifier.val_step,
            ['backbone', 'softmax_cross_entropy']
        )
        self.assertNotCalling(
            ans.classification.AutogradClassifier.val_step,
            ['zero_grad', 'backprop', 'step']
        )

    def test_train_step(self):
        self.model.train()
        expected_losses = torch.tensor([1.88818, 1.60737])
        expected_linear_1_weights = [
            torch.tensor([
                [ 0.2421,  0.3521, -0.2181,  0.0204, -0.2764, -0.1672],
                [ 0.1586, -0.4182,  0.2404, -0.3126,  0.0325, -0.4449],
                [-0.1920, -0.0014, -0.2452, -0.1091, -0.0894,  0.1670],
                [ 0.4535,  0.0890,  0.1553,  0.2837,  0.2994, -0.1221]
            ]),
            torch.tensor([
                [ 0.2421,  0.3521, -0.2182,  0.0204, -0.2764, -0.1672],
                [ 0.1583, -0.4185,  0.2405, -0.3125,  0.0325, -0.4449],
                [-0.1924, -0.0020, -0.2451, -0.1089, -0.0894,  0.1671],
                [ 0.4532,  0.0887,  0.1555,  0.2838,  0.2995, -0.1221]
            ])
        ]
        for i in range(2):
            loss, logits = self.model.train_step(self.inputs, self.targets)
            self.assertIsInstance(loss, float)
            self.assertIsInstance(logits, torch.Tensor)
            self.assertTensorsClose(loss, expected_losses[i])
            self.assertTensorsClose(self.linear_1.weight.data, expected_linear_1_weights[i])
    
    def test_val_step(self):
        self.model.eval()
        expected_losses = torch.tensor([1.88818, 1.88818])
        expected_params = [p.data.clone() for p in self.model.parameters()]
        for i in range(2):
            loss, logits = self.model.val_step(self.inputs, self.targets)
            self.assertIsInstance(loss, float)
            self.assertIsInstance(logits, torch.Tensor)
            self.assertTensorsClose(loss, expected_losses[i])
            for p, ep in zip(self.model.parameters(), expected_params):
                self.assertTensorsClose(p.data, ep)


class TestTrainEpochValidate(ANSTestCase):

    def test_implementation(self):
        self.assertCalling(ans.classification.train_epoch, ['train'])
        self.assertNotCalling(ans.classification.train_epoch, ['eval'])
        self.assertNotCalling(ans.classification.validate, ['train'])
        self.assertCalling(ans.classification.validate, ['eval'])

    def test_train_epoch(self):
        pass

    def test_validate(self):
        pass


class TestBatchLoaderWithDevice(TestBatchLoader):

    devices = [
        'cpu',
        'meta'
    ]

    def test_device(self):
        for device in self.devices:
            for dataset in [self.unsupervised_dataset, self.supervised_dataset]:
                loader = ans.data.BatchLoader(dataset, device=device)
                batch = next(iter(loader))
                for tensor in batch:
                    self.assertEqual(tensor.device, torch.device(device))


class TestDataPreprocessor(ANSTestCase):

    def check_dataset(self, orig_ds, prep_ds, train=False):
        x_orig = orig_ds.data
        x_prep, y_prep = prep_ds.tensors
        self.assertEqual(x_prep.ndim, 2, msg='Preprocessed dataset must be 2D matrix (N ,D)')
        if not train:
            self.assertEqual(len(orig_ds), len(prep_ds), msg='Augmentation not allowed in validation set')
        self.assertEqual(x_prep.size(1), x_orig[0].size, msg='Resizing and/or feature extraction not allowed')
        self.assertEqual(y_prep.size(0), x_prep.size(0))
        self.assertEqual(y_prep.ndim, 1, msg='Targets must be a vector of integers')
    
    def test_preprocess(self):
        train_dataset = torchvision.datasets.CIFAR10(
            root = '../data',
            train = True,
        )
        preprocessor = self.params['preprocessor_cls']()
        preprocessor.fit(train_dataset)
        self.check_dataset(train_dataset, preprocessor.transform(train_dataset, train=True), train=True)
        val_dataset = torchvision.datasets.CIFAR10(
            root = '../data',
            train = False,
        )
        self.check_dataset(val_dataset, preprocessor.transform(val_dataset, train=False), train=False)


class TestValAccuracy45(ANSTestCase):

    def __init__(self, methodName: str = '', **params):
        super().__init__(methodName, **params)
        self.min_val_acc = 0.45

    def test_val_acc(self):
        model = ans.classification.AutogradClassifier.load('../output/autograd_classifier.pt')
        model.to(device='cpu')
        
        train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        val_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)
        
        preprocessor = self.params['preprocessor_cls']()
        preprocessor.fit(train_dataset)
        train_dataset = preprocessor.transform(train_dataset, train=True)
        val_dataset = preprocessor.transform(val_dataset, train=False)

        train_loader = ans.data.BatchLoader(train_dataset, batch_size=100, shuffle=True)
        val_loader = ans.data.BatchLoader(val_dataset, batch_size=100, shuffle=False)
        
        train_loss, train_acc = ans.classification.validate(model, train_loader)
        val_loss, val_acc = ans.classification.validate(model, val_loader)
        self.assertGreaterEqual(val_acc, self.min_val_acc)
        self.assertLess(val_acc, train_acc + 0.01)
        self.assertGreater(val_loss, train_loss - 0.1)


class TestValAccuracy55(TestValAccuracy45):

    def __init__(self, methodName: str = '', **params):
        super().__init__(methodName, **params)
        self.min_val_acc = 0.55


class TestValAccuracy60(TestValAccuracy45):

    def __init__(self, methodName: str = '', **params):
        super().__init__(methodName, **params)
        self.min_val_acc = 0.60
