from typing import Any, Protocol, Self

import torch

import ans


class ClassificationModel(Protocol):
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training: forward pass, loss computation, backward pass, parameters update.

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ...

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Performs one step of validation: forward pass, loss computation.

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ...


class LinearSoftmaxModel:
    def __init__(self, in_size: int, out_size: int, weight_scale: float = 1e-3) -> None:
        """
        Args:
            in_size: input dimension D
            out_size: number of classes K
            weight_scale: standard deviation of the normal distribution or a bound U(-weight_scale, weight_scale)
                          of uniform distribution used to initialize the weights
        """
        ########################################
        # TODO: implement

        self.weight = ...
        self.bias = ...

        ########################################

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training of linear softmax cross entropy classifier:
        - s = w * x + b
        - loss = cross_entropy(s, y)
        - compute gradients dloss/dw, dloss/db
        - params = params - learning_rate * dloss/dparams (SGD update)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average cross entropy loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Performs one step of validation of linear softmax cross entropy classifier:
        - s = w * x + b
        - loss = cross_entropy(s, y)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average cross entropy loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits


class LinearSVMModel(LinearSoftmaxModel):
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training of linear support vector machine (SVM) classifier:
        - s = w * x + b
        - loss = hinge_loss(s, y)
        - compute gradients dloss/dw, dloss/db
        - params = params - learning_rate * dloss/dparams (SGD update)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average hinge loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Performs one step of validation of linear support vector machine (SVM) classifier:
        - s = w * x + b
        - loss = hinge_loss(s, y)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average hinge loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits


class TwoLayerPerceptron:
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        weight_scale: float = 0.001,
    ) -> None:
        self.weight1 = ...
        self.bias1 = ....
        self.weight2 = ...
        self.bias2 = ...

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement
        #
        # Some of the intermediate gradients that are to be expected for the test data used in
        # test_two_layer_perceptron.TestTrainStep:
        #   dlogits = torch.tensor([  # gradient w.r.t. output of the network
        #     [ 0.0072,  0.0924, -0.0996],
        #     [-0.0484,  0.0199,  0.0285],
        #     [ 0.0047, -0.0258,  0.0211],
        #     [ 0.0126,  0.0778, -0.0904],
        #     [ 0.0333, -0.0696,  0.0362],
        #     [ 0.0696,  0.0257, -0.0953],
        #     [-0.0290,  0.0066,  0.0223],
        #     [ 0.1003,  0.0059, -0.1062]
        #  ])
        #  dhidden = torch.tensor([  # gradient w.r.t. output of sigmoid
        #    [ 0.1521,  0.0635, -0.1338, -0.0515],
        #    [ 0.0342,  0.0584, -0.1164,  0.0261],
        #    [-0.0427, -0.0237,  0.0491,  0.0094],
        #    [ 0.1279,  0.0477, -0.1013, -0.0482],
        #    [-0.1156, -0.0825,  0.1688,  0.0099],
        #    [ 0.0404, -0.0429,  0.0813, -0.0646],
        #    [ 0.0118,  0.0310, -0.0613,  0.0182],
        #    [ 0.0070, -0.0853,  0.1665, -0.0776]
        #  ])
        #  dprehidden = torch.tensor([  # gradient w.r.t. output of the first linear layer (before sigmoid)
        #    [ 0.0338,  0.0038, -0.0153, -0.0002],
        #    [ 0.0047,  0.0002, -0.0291,  0.0004],
        #    [-0.0092, -0.0003,  0.0007,  0.0001],
        #    [ 0.0315,  0.0097, -0.0107, -0.0039],
        #    [-0.0284, -0.0081,  0.0328,  0.0020],
        #    [ 0.0091, -0.0002,  0.0203, -0.0102],
        #    [ 0.0003,  0.0007, -0.0072,  0.0028],
        #    [ 0.0003, -0.0001,  0.0153, -0.0004]])
        #  ])

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def save(self, filename: str) -> None:
        torch.save(dict(weight1=self.weight1, bias1=self.bias1, weight2=self.weight2, bias2=self.bias2), filename)

    @classmethod
    def load(cls, filename: str) -> Self:
        dic = torch.load(filename, weights_only=True)
        model = cls(*dic['weight1'].shape, dic['weight2'].shape[1])
        model.weight1 = dic['weight1']
        model.bias1 = dic['bias1']
        model.weight2 = dic['weight2']
        model.bias2 = dic['bias2']
        return model


class TwoLayerPerceptronAutograd:
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        weight_scale: float = 0.001,
    ) -> None:
        ########################################
        # TODO: implement

        self.weight1 = ...
        self.bias1 = ...
        self.weight2 = ...
        self.bias2 = ...

        # ENDTODO
        ########################################

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def save(self, filename: str) -> None:
        torch.save(
            dict(weight1=self.weight1.data, bias1=self.bias1.data, weight2=self.weight2.data, bias2=self.bias2.data),
            filename,
        )

    @classmethod
    def load(cls, filename: str) -> Self:
        dic = torch.load(filename, weights_only=True)
        model = cls(*dic['weight1'].shape, dic['weight2'].shape[1])
        model.weight1 = ans.autograd.Variable(dic['weight1'])
        model.bias1 = ans.autograd.Variable(dic['bias1'])
        model.weight2 = ans.autograd.Variable(dic['weight2'])
        model.bias2 = ans.autograd.Variable(dic['bias2'])
        return model


class AutogradClassifier(ans.nn.Module):
    def __init__(self, backbone: ans.nn.Module, optimizer: ans.nn.Optimizer) -> None:
        self.backbone = backbone
        self.optimizer = optimizer

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def save(self, filename: str) -> None:
        torch.save(self, filename)  # not the correct & safe way to do this

    @classmethod
    def load(cls, filename: str) -> Self:
        return torch.load(filename, weights_only=False)  # not the correct & safe way to do this


def accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Args:
        scores: output linear scores (logits or probabilities); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        acc: average accuracy on the batch; single number (scalar), e.g. 0.364
    """

    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################

    return acc


def softmax_cross_entropy(
    logits: torch.Tensor | ans.autograd.Variable, targets: torch.Tensor
) -> torch.Tensor | ans.autograd.Variable:
    """
    Args:
        scores: output linear scores (logits or probabilities); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        loss: average loss on the batch; single number (scalar), e.g. 2.31
    """
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################


def train_epoch(model: ClassificationModel, loader: ans.data.BatchLoader, **train_step_kwargs) -> Any:
    """
    Trains `model` on the dataset `loader` for one epoch.

    Args:
        model: Model to train. Must have method `train_step`.
        loader: loader of the training dataset
    Returns:
        return whatever is needed
    """
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################


def validate(model: ClassificationModel, loader: ans.data.BatchLoader) -> tuple[float, float]:
    """
    Validates `model` on the dataset `loader`.

    Args:
        model: Model to be validated. Must have method `val_step`.
        loader: loader of the training dataset
    Returns:
        mean_loss: average loss achieved on the dataset during training
        mean_acc: average accuracy achieved on the dataset during model training
    """
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################
