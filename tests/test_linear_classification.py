import numpy as np
import PIL.Image
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


class TestBatchLoader(ANSTestCase):

    def setUp(self) -> None:
        self.x = torch.tile(torch.arange(10)[:, None], (1, 2))
        self.y = torch.arange(10)
        self.unsupervised_dataset = torch.utils.data.TensorDataset(self.x)
        self.supervised_dataset = torch.utils.data.TensorDataset(self.x, self.y)

    def test_output_tuple_unsupervised(self):
        loader = ans.data.BatchLoader(self.unsupervised_dataset)
        batch = next(iter(loader))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 1)

    def test_output_tuple_supervised(self):
        loader = ans.data.BatchLoader(self.supervised_dataset)
        batch = next(iter(loader))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)

    def test_defaults(self) -> None:
        loader = ans.data.BatchLoader(self.supervised_dataset)
        xb, yb = next(iter(loader))
        self.assertTupleEqual(xb.shape, self.x.shape)
        self.assertTupleEqual(yb.shape, self.y.shape)
        self.assertTensorsClose(xb, self.x)
        self.assertTensorsClose(yb, self.y)

    def test_batch_size_even(self) -> None:
        batch_size = 2
        loader = ans.data.BatchLoader(self.supervised_dataset, batch_size=batch_size)
        batches = [(xb, yb) for xb, yb in loader]
        self.assertEqual(len(batches), len(self.x) // batch_size)
        self.assertEqual(sum(len(xb) for xb, yb in batches), len(self.x))
        expected_batch_sizes = [batch_size] * (len(self.x) // batch_size)
        self.assertListEqual([len(xb) for xb, yb in batches], expected_batch_sizes)
        self.assertListEqual([len(yb) for xb, yb in batches], expected_batch_sizes)

    def test_batch_size_uneven(self) -> None:
        batch_size = 3
        loader = ans.data.BatchLoader(self.supervised_dataset, batch_size=batch_size)
        batches = [(xb, yb) for xb, yb in loader]
        self.assertEqual(len(batches), -(-len(self.x) // batch_size))  # -(-a//b)) ... ceil
        self.assertEqual(sum(len(xb) for xb, yb in batches), len(self.x))
        expected_batch_sizes = [batch_size] * (len(self.x) // batch_size)
        if sum(expected_batch_sizes) < len(self.x):
            expected_batch_sizes.append(len(self.x) - sum(expected_batch_sizes))
        self.assertListEqual([len(xb) for xb, yb in batches], expected_batch_sizes)
        self.assertListEqual([len(yb) for xb, yb in batches], expected_batch_sizes)

    def test_shuffle(self):
        loader = ans.data.BatchLoader(self.supervised_dataset, shuffle=True)
        xb, yb = next(iter(loader))
        self.assertFalse(torch.all(xb == self.x))
        self.assertFalse(torch.all(yb == self.y))
        self.assertTensorsClose(xb, self.x[xb[:, 0]])
        self.assertTensorsClose(yb, self.y[xb[:, 0]])


class TestInit(ANSTestCase):

    def test_init_params(self):
        D, K, a = 1000, 50, 0.0001234
        model = ans.classification.LinearSoftmaxModel(D, K, weight_scale=a)
        self.assertIsInstance(model.weight, torch.Tensor)
        self.assertIsInstance(model.bias, torch.Tensor)
        self.assertTupleEqual(model.weight.shape, (D, K))
        self.assertTupleEqual(model.bias.shape, (K,))
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(model.bias.dtype, torch.float32)
        self.assertTensorsClose(torch.mean(torch.abs(model.weight)), torch.tensor(0.75 * a), rtol=1/3)  # TODO: improve


class TestTrainStepSoftmax(ANSTestCase):

    def setUp(
        self,
        models_cls: type[ans.classification.LinearSoftmaxModel] = ans.classification.LinearSoftmaxModel
    ) -> None:
        self.model_cls = models_cls
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
        self.model = self.model_cls(5, 3)
        self.model.weight = torch.tensor([
            [ 0.4935, -0.2807, -0.6546],
            [ 1.2959, -0.0452,  0.1982],
            [ 0.0118, -0.2656,  0.8765],
            [-0.9573, -1.6125, -0.4814],
            [ 2.1671,  0.5214, -0.0168]
        ])
        self.model.bias = torch.tensor([-0.3000, -0.0143,  0.3636])
        self.learning_rate = 0.1234

    def test_implementation(self):
        self.assertNoLoops(ans.classification.LinearSoftmaxModel.train_step)
        self.assertNotCalling(
            ans.classification.LinearSoftmaxModel.train_step,
            ['CrossEntropyLoss', 'cross_entropy', 'Softmax', 'softmax', 'backward']
        )

    def test_step(self):
        loss, scores = self.model.train_step(self.inputs, self.targets, learning_rate=self.learning_rate)
        expected_loss = 1.5676
        self.assertTensorsClose(loss, expected_loss)
        expected_scores = torch.tensor([
            [ 0.3274, -1.2675, -1.5678],
            [-0.4752,  1.9232,  2.3318],
            [ 1.7525, -0.8438, -1.6759],
            [ 0.0321, -1.1952, -1.0957],
            [-0.7454,  0.8092,  0.5695],
            [-3.2430, -0.8081,  0.3545],
            [-0.0447, -1.5828,  1.1300],
            [-1.2953,  0.8460,  2.4296]
        ])
        self.assertTensorsClose(scores, expected_scores)
        expected_weight = torch.tensor([
            [ 0.4145, -0.2273, -0.6290],
            [ 1.2708, -0.0422,  0.2203],
            [ 0.0637, -0.2696,  0.8286],
            [-0.9776, -1.6148, -0.4588],
            [ 2.1740,  0.5238, -0.0262]
        ])
        self.assertTensorsClose(self.model.weight, expected_weight)
        expected_bias = torch.tensor([-0.3103, -0.0102,  0.3698])
        self.assertTensorsClose(self.model.bias, expected_bias)


class TestValStepSoftmax(TestTrainStepSoftmax):

    def test_implementation(self):
        self.assertNoLoops(ans.classification.LinearSoftmaxModel.val_step)
        self.assertNotCalling(
            ans.classification.LinearSoftmaxModel.val_step,
            ['CrossEntropyLoss', 'cross_entropy', 'Softmax', 'softmax', 'backward', 'HingeEmbeddingLoss',
             'hinge_embedding_loss', 'MultiMarginLoss', 'multi_margin_loss']
        )

    def test_step(self):
        expected_weight = self.model.weight.clone()  # sholud not be modified in val_step
        expected_bias = self.model.bias.clone()  # dtto
        loss, scores = self.model.val_step(self.inputs, self.targets)
        expected_loss = 1.5676  # same as train_step
        self.assertTensorsClose(loss, expected_loss)
        expected_scores = torch.tensor([  # same as train_step
            [ 0.3274, -1.2675, -1.5678],
            [-0.4752,  1.9232,  2.3318],
            [ 1.7525, -0.8438, -1.6759],
            [ 0.0321, -1.1952, -1.0957],
            [-0.7454,  0.8092,  0.5695],
            [-3.2430, -0.8081,  0.3545],
            [-0.0447, -1.5828,  1.1300],
            [-1.2953,  0.8460,  2.4296]
        ])
        self.assertTensorsClose(scores, expected_scores)
        self.assertTensorsClose(self.model.weight, expected_weight)
        self.assertTensorsClose(self.model.bias, expected_bias)


class TestAccuracy(ANSTestCase):

    def setUp(self) -> None:
        self.scores = torch.tensor([
            [0.6803, 0.5091, 0.9956, 0.3223, 0.0412],
            [0.2676, 0.8497, 0.8004, 0.8800, 0.1041],
            [0.3104, 0.8930, 0.3836, 0.4640, 0.4453],
            [0.0873, 0.0481, 0.6611, 0.3160, 0.0896],
            [0.0683, 0.7224, 0.0070, 0.7719, 0.3652],
            [0.6974, 0.0702, 0.0464, 0.2236, 0.6874],
            [0.9554, 0.3023, 0.1847, 0.4373, 0.4032],
            [0.2513, 0.4600, 0.4850, 0.4161, 0.7903],
            [0.4853, 0.6241, 0.5177, 0.9216, 0.5183],
            [0.3431, 0.0639, 0.1924, 0.2636, 0.3692],
            [0.4227, 0.7406, 0.8387, 0.2885, 0.5232],
            [0.2162, 0.3309, 0.6583, 0.0697, 0.7484],
            [0.1898, 0.5281, 0.0541, 0.3052, 0.4897],
            [0.2559, 0.3936, 0.4042, 0.6656, 0.9526],
            [0.5814, 0.6609, 0.7857, 0.7971, 0.3776],
            [0.6371, 0.2196, 0.8141, 0.8793, 0.7719],
            [0.6085, 0.6812, 0.0712, 0.2430, 0.0683],
            [0.0146, 0.3447, 0.9563, 0.7547, 0.0045],
            [0.4896, 0.8539, 0.5926, 0.2179, 0.6305]
        ])
        self.targets = torch.tensor([1, 3, 3, 3, 0, 2, 4, 3, 2, 1, 0, 1, 1, 2, 0, 4, 1, 2, 2])

    def test_implementation(self):
        self.assertNotCalling(ans.classification.accuracy, ['accuracy'])
        self.assertNoLoops(ans.classification.accuracy)

    def test_accuracy(self):
        acc = ans.classification.accuracy(self.scores, self.targets)
        expected_acc = torch.tensor(0.2105)
        self.assertTensorsClose(torch.tensor(acc), expected_acc)


class TestTrainEpoch(ANSTestCase):

    def setUp(self) -> None:
        x = torch.tensor([
            [0.6803, 0.5091, 0.9956, 0.3223, 0.0412],
            [0.2676, 0.8497, 0.8004, 0.8800, 0.1041],
            [0.3104, 0.8930, 0.3836, 0.4640, 0.4453],
            [0.0873, 0.0481, 0.6611, 0.3160, 0.0896],
            [0.0683, 0.7224, 0.0070, 0.7719, 0.3652],
            [0.6974, 0.0702, 0.0464, 0.2236, 0.6874],
            [0.9554, 0.3023, 0.1847, 0.4373, 0.4032],
            [0.2513, 0.4600, 0.4850, 0.4161, 0.7903],
            [0.4853, 0.6241, 0.5177, 0.9216, 0.5183],
            [0.3431, 0.0639, 0.1924, 0.2636, 0.3692],
            [0.4227, 0.7406, 0.8387, 0.2885, 0.5232]
        ])
        y = torch.tensor([2, 2, 0, 1, 2, 1, 2, 0, 2, 0, 2])
        self.dataset = torch.utils.data.TensorDataset(x, y)
        self.loader = ans.data.BatchLoader(self.dataset, batch_size=2)
        self.model = ans.classification.LinearSoftmaxModel(5, 3)
        self.model.weight = torch.tensor([
            [ 0.5258,  0.7127, -0.4763],
            [-0.6679,  1.6683,  0.2156],
            [ 1.9647,  0.4357, -1.1398],
            [-0.5301,  0.0127, -0.2765],
            [ 1.4234, -0.8227,  0.7401]
        ])
        self.model.bias = torch.tensor([0.2106, 1.1809, 0.5028])

    def test_implementation(self) -> None:
        self.assertCalling(ans.classification.train_epoch, ['train_step'])

    def test_step(self) -> None:
        ans.classification.train_epoch(self.model, self.loader, learning_rate=0.1234)
        expected_weight = torch.tensor([
            [ 0.4746,  0.5978, -0.3102],
            [-0.6881,  1.4519,  0.4522],
            [ 1.9002,  0.2733, -0.9130],
            [-0.5571, -0.1583, -0.0785],
            [ 1.4112, -0.9187,  0.8483]
        ])
        self.assertTensorsClose(self.model.weight, expected_weight)
        expected_bias = torch.tensor([0.1476, 0.9072, 0.8395])
        self.assertTensorsClose(self.model.bias, expected_bias)


class TestValidate(TestTrainEpoch):

    def test_implementation(self) -> None:
        self.assertCalling(ans.classification.validate, ['val_step'])

    def test_step(self) -> None:
        mean_loss, mean_acc = ans.classification.validate(self.model, self.loader)
        expected_weight = torch.tensor([
            [ 0.5258,  0.7127, -0.4763],
            [-0.6679,  1.6683,  0.2156],
            [ 1.9647,  0.4357, -1.1398],
            [-0.5301,  0.0127, -0.2765],
            [ 1.4234, -0.8227,  0.7401]
        ])
        self.assertTensorsClose(self.model.weight, expected_weight)  # model params must not change
        expected_bias = torch.tensor([0.2106, 1.1809, 0.5028])
        self.assertTensorsClose(self.model.bias, expected_bias)    # model params must not change
        expected_loss = 2.162376
        self.assertTensorsClose(mean_loss, expected_loss)
        expected_acc = 0.1818182
        self.assertTensorsClose(mean_acc, expected_acc)


class TestSoftmaxValAccuracy(ANSTestCase):

    @staticmethod
    def load_model(
        path: str,
        cls: type[ans.classification.LinearSoftmaxModel] = ans.classification.LinearSoftmaxModel
    ) -> ans.classification.LinearSoftmaxModel:
        dic = torch.load(path, weights_only=True)
        model = cls(*dic['weight'].shape)
        model.weight = dic['weight']
        model.bias = dic['bias']
        return model

    def test_softmax_val_acc(self):
        model = self.load_model('../output/linear_classification_softmax_weights.pt')
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
        self.assertGreaterEqual(val_acc, 0.4)
        self.assertGreater(val_loss, train_loss - 0.1)
        self.assertLess(val_acc, train_acc + 0.01)


class TestTrainStepSVM(TestTrainStepSoftmax):

    def setUp(self) -> None:
        super().setUp(models_cls=ans.classification.LinearSVMModel)
        self.model.weight = torch.tensor([
            [-2.4231,  0.9126, -0.8539],
            [ 1.5711,  1.1129,  0.2845],
            [ 1.1680,  0.5390, -0.9191],
            [ 0.8099, -0.3242, -0.0966],
            [ 1.1331, -0.6054,  0.3497]
        ])
    
    def test_implementation(self):
        self.assertNoLoops(ans.classification.LinearSVMModel.train_step)
        self.assertNotCalling(
            ans.classification.LinearSVMModel.train_step,
            ['backward', 'HingeEmbeddingLoss', 'hinge_embedding_loss', 'MultiMarginLoss', 'multi_margin_loss']
        )

    def test_step(self):
        loss, scores = self.model.train_step(self.inputs, self.targets, learning_rate=self.learning_rate)
        expected_loss = 1.16139
        self.assertTensorsClose(loss, expected_loss)
        expected_scores = torch.tensor([
            [-2.5908,  1.2930,  0.6727],
            [ 0.4570, -2.0349, -0.5554],
            [-7.4694,  1.8186, -2.3728],
            [-2.3457, -0.1554,  0.0126],
            [-1.8427, -0.3405,  0.0509],
            [ 0.4428, -2.2182,  0.8571],
            [ 3.5970, -0.2432, -0.2476],
            [ 5.2671, -3.1015,  1.4054]
        ])
        self.assertTensorsClose(scores, expected_scores)
        expected_weight = torch.tensor([
            [-2.3737,  0.8821, -0.8729],
            [ 1.6034,  1.0920,  0.2731],
            [ 1.1563,  0.5669, -0.9354],
            [ 0.8009, -0.3556, -0.0562],
            [ 1.1266, -0.6100,  0.3608]
        ])
        self.assertTensorsClose(self.model.weight, expected_weight)
        expected_bias = torch.tensor([-0.3309, -0.0297,  0.4099])
        self.assertTensorsClose(self.model.bias, expected_bias)


class TestValStepSVM(TestTrainStepSVM):

    def test_implementation(self):
        self.assertNoLoops(ans.classification.LinearSVMModel.val_step)
        self.assertNotCalling(
            ans.classification.LinearSVMModel.val_step,
            ['backward', 'HingeEmbeddingLoss', 'hinge_embedding_loss', 'MultiMarginLoss', 'multi_margin_loss']
        )

    def test_step(self):
        expected_weight = self.model.weight.clone()  # sholud not be modified in val_step
        expected_bias = self.model.bias.clone()  # dtto
        loss, scores = self.model.val_step(self.inputs, self.targets)
        expected_loss = 1.16139  # same as train step
        self.assertTensorsClose(loss, expected_loss)
        expected_scores = torch.tensor([  # same as train_step
            [-2.5908,  1.2930,  0.6727],
            [ 0.4570, -2.0349, -0.5554],
            [-7.4694,  1.8186, -2.3728],
            [-2.3457, -0.1554,  0.0126],
            [-1.8427, -0.3405,  0.0509],
            [ 0.4428, -2.2182,  0.8571],
            [ 3.5970, -0.2432, -0.2476],
            [ 5.2671, -3.1015,  1.4054]
        ])
        self.assertTensorsClose(scores, expected_scores)
        self.assertTensorsClose(self.model.weight, expected_weight)
        self.assertTensorsClose(self.model.bias, expected_bias)


class TestSVMValAccuracy(ANSTestCase):

    def test_svm_val_acc(self):
        model = TestSoftmaxValAccuracy.load_model(
            '../output/linear_classification_svm_weights.pt',
            cls = ans.classification.LinearSVMModel
        )
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
        self.assertGreaterEqual(val_acc, 0.39)
        self.assertGreater(val_loss, train_loss - 0.1)
        self.assertLess(val_acc, train_acc + 0.01)
