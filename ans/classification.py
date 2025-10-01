from typing import Any, Protocol

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

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
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

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
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

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
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


def softmax_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
