import numpy as np
import PIL
import torch
import torchvision

import ans
from tests import ANSTestCase


class TestPreprocess(ANSTestCase):
    def test_preprocess(self):
        img = PIL.Image.fromarray(
            np.array(
                [[[173, 129, 253],
                  [ 82,  10,  68]],
                 [[216, 204, 224],
                  [ 26,  79, 227]]],
                dtype = np.uint8
            )
        )
        x = self.params['preprocess_fn'](img)
        expected_x = torch.tensor([
            0.6784, 0.5059, 0.9922, 0.3216, 0.0392, 0.2667,
            0.8471, 0.8000, 0.8784, 0.1020, 0.3098, 0.8902
        ])
        self.assertTensorsClose(x, expected_x)


class TestInit(ANSTestCase):
    def check_init_layer(self, w: torch.Tensor, b: torch.Tensor, expected_shape: tuple[int, int], init_scale: float):
        D, K = expected_shape
        self.assertIsInstance(w, torch.Tensor)
        self.assertIsInstance(b, torch.Tensor)
        self.assertTupleEqual(w.shape, (D, K))
        self.assertTupleEqual(b.shape, (K,))
        self.assertEqual(w.dtype, torch.float32)
        self.assertEqual(b.dtype, torch.float32)
        self.assertTensorsClose(torch.mean(torch.abs(w)), torch.tensor(0.75 * init_scale), rtol=1/3)  # TODO: improve

    def test_init_params(self):
        D, H, K, a = 1000, 500, 50, 0.0001234
        model = ans.classification.TwoLayerPerceptron(D, H, K, weight_scale=a)
        self.check_init_layer(model.weight1, model.bias1, (D, H), a)
        self.check_init_layer(model.weight2, model.bias2, (H, K), a)


class TestTrainStep(ANSTestCase):
    def setUp(self) -> None:
        self.inputs = torch.tensor([
            [ 1.1008,  0.9469, -1.2206,  0.6903, -0.2158],
            [-0.7691, -1.8333,  1.5317, -1.0340,  0.7255],
            [ 3.0616, -0.5493,  0.1936,  0.1769,  0.6555],
            [ 0.9366, -0.2081, -0.4776,  0.7880,  0.4151],
            [ 0.0620, -0.6139,  0.1135, -0.5547, -0.0982],
            [-0.9524, -1.0680, -0.1334,  0.6390, -0.2195],
            [-0.3594,  0.1379,  1.1483,  1.0245,  0.5635],
            [-2.2524, -1.0259,  0.8902, -0.0527,  0.6390]
        ])
        self.targets = torch.tensor([2, 0, 1, 2, 1, 2, 0, 2])
        self.model = ans.classification.TwoLayerPerceptron(5, 4, 3)
        self.model.weight1 = torch.tensor([
            [ 0.6404,  1.8266, -1.1524,  1.6550],
            [-0.6438,  1.1360,  0.8619,  1.1973],
            [-1.2408, -1.6897,  1.2341, -1.0806],
            [-0.7942, -0.7537,  0.4722, -0.4640],
            [-0.9144,  1.8758, -0.4184, -1.1447]
        ])
        self.model.bias1 = torch.tensor([-0.5695, -1.5478, -0.3433,  1.3487])
        self.model.weight2 = torch.tensor([
            [ 0.4080,  2.0843,  0.4360],
            [-1.4321,  0.2203, -0.5365],
            [ 1.9807, -1.3583,  0.2267],
            [ 0.4421,  0.6868,  1.1862]
        ])
        self.model.bias2 = torch.tensor([-1.3626, -1.2764, -1.4689])
        self.learning_rate = 0.1234

    def test_implementation(self):
        self.assertNoLoops(ans.classification.TwoLayerPerceptron.train_step)
        self.assertNotCalling(
            ans.classification.TwoLayerPerceptron.train_step,
            ['CrossEntropyLoss', 'cross_entropy', 'Softmax', 'softmax', 'backward']
        )

    def test_step(self):
        loss, scores = self.model.train_step(self.inputs, self.targets, learning_rate=self.learning_rate)
        expected_loss = 1.000912
        self.assertTensorsClose(loss, expected_loss)
        expected_scores = torch.tensor([
            [-1.7295,  0.8228, -0.4691],
            [-0.2736, -1.6215, -1.2637],
            [-2.0301,  1.0253, -0.5212],
            [-1.5671,  0.2571, -0.5536],
            [-0.4472,  0.0605, -0.3639],
            [-0.1209, -1.1177, -0.9706],
            [ 0.4106, -2.2611, -1.0479],
            [ 0.4364, -2.3934, -1.2390]
        ])
        self.assertTensorsClose(scores, expected_scores)
        expected_weight1 = torch.tensor([
            [ 0.6375,  1.8252, -1.1460,  1.6543],
            [-0.6474,  1.1352,  0.8641,  1.1960],
            [-1.2340, -1.6886,  1.2359, -1.0815],
            [-0.8020, -0.7556,  0.4725, -0.4630],
            [-0.9149,  1.8753, -0.4155, -1.1450]
        ])
        self.assertTensorsClose(self.model.weight1, expected_weight1)
        expected_bias1 = torch.tensor([-0.5747, -1.5485, -0.3441,  1.3499])
        self.assertTensorsClose(self.model.bias1, expected_bias1)
        expected_weight2 = torch.tensor([
            [ 0.4015,  2.0780,  0.4488],
            [-1.4350,  0.2068, -0.5201],
            [ 1.9699, -1.3629,  0.2421],
            [ 0.4353,  0.6752,  1.2046]
        ])
        self.assertTensorsClose(self.model.weight2, expected_weight2)
        expected_bias2 = torch.tensor([-1.3811, -1.2928, -1.4339])
        self.assertTensorsClose(self.model.bias2, expected_bias2)


class TestValStep(TestTrainStep):
    def test_implementation(self):
        self.assertNoLoops(ans.classification.TwoLayerPerceptron.val_step)
        self.assertNotCalling(
            ans.classification.TwoLayerPerceptron.val_step,
            ['CrossEntropyLoss', 'cross_entropy', 'Softmax', 'softmax', 'backward']
        )

    def test_step(self):
        expected_weight1 = self.model.weight1.clone()
        expected_bias1 = self.model.bias1.clone()
        expected_weight2 = self.model.weight2.clone()
        expected_bias2 = self.model.bias2.clone()
        loss, scores = self.model.val_step(self.inputs, self.targets)
        expected_loss = 1.000912
        self.assertTensorsClose(loss, expected_loss)
        expected_scores = torch.tensor([
            [-1.7295,  0.8228, -0.4691],
            [-0.2736, -1.6215, -1.2637],
            [-2.0301,  1.0253, -0.5212],
            [-1.5671,  0.2571, -0.5536],
            [-0.4472,  0.0605, -0.3639],
            [-0.1209, -1.1177, -0.9706],
            [ 0.4106, -2.2611, -1.0479],
            [ 0.4364, -2.3934, -1.2390]
        ])
        self.assertTensorsClose(scores, expected_scores)
        self.assertTensorsClose(self.model.weight1, expected_weight1)
        self.assertTensorsClose(self.model.bias1, expected_bias1)
        self.assertTensorsClose(self.model.weight2, expected_weight2)
        self.assertTensorsClose(self.model.bias2, expected_bias2)


class TestValAccuracy45(ANSTestCase):
    def __init__(self, methodName: str = '', **params):
        super().__init__(methodName, **params)
        self.min_val_acc = 0.45

    def test_val_acc(self):
        model = ans.classification.TwoLayerPerceptron.load('../output/two_layer_perceptron_weights.pt')
        train_dataset = torchvision.datasets.CIFAR10(
            root = '../data',
            train = True,
            transform = self.params['preprocess_fn']
        )
        train_loader = ans.data.BatchLoader(
            train_dataset,
            batch_size = 100
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root = '../data',
            train = False,
            transform = self.params['preprocess_fn']
        )
        val_loader = ans.data.BatchLoader(
            val_dataset,
            batch_size = 100
        )
        train_loss, train_acc = ans.classification.validate(model, train_loader)
        val_loss, val_acc = ans.classification.validate(model, val_loader)
        self.assertGreaterEqual(val_acc, self.min_val_acc)
        self.assertLess(val_acc, train_acc)
        self.assertGreater(val_loss, train_loss)


class TestValAccuracy50(TestValAccuracy45):
    def __init__(self, methodName: str = '', **params):
        super().__init__(methodName, **params)
        self.min_val_acc = 0.50
