import numpy as np
from typing import Optional, Callable
from uuid import uuid4

def sum_broadcasted_axes(inp_shape: tuple[int], out: np.ndarray) -> np.ndarray:
    """
    Sums over axes that were broadcasted while converting inp into out.
    """ 
    if inp_shape == out.shape:
        return out

    # First sum over any dimensions that don't appear in the input
    axes = list(range(len(out.shape) - len(inp_shape)))
    if len(axes) > 0:
        out = np.sum(out, axis=tuple(axes))

    # Now sum over the axes that were broadcast
    axes = []
    for i in range(len(inp_shape)):
        if inp_shape[i] != out.shape[i]:
            assert inp_shape[i] == 1 # This should be trivially true
            axes.append(i)

    if len(axes) > 0:
        out = np.sum(out, axis=tuple(axes), keepdims=True)
    
    return out


### Several common backward operations
def transpose_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for inp[0].T.
    """
    assert len(inp) == 1
    
    inp[0].backward(grad.T)

def neg_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for -inp[0]
    """
    assert len(inp) == 1

    inp[0].backward(-grad)

def add_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for inp[0] + inp[1].
    """
    assert len(inp) == 2, "Add backward requires two elements"

    # Handle broadcasting
    inp[0].backward(grad)
    inp[1].backward(grad)

def mul_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for inp[0] * inp[1]
    """
    assert len(inp) == 2
    
    inp[0].backward(grad * inp[1].data)
    inp[1].backward(grad * inp[0].data)

def matmul_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for inp[0] @ inp[1]
    """
    assert len(inp) == 2

    inp[0].backward(grad @ inp[1].data.T)
    inp[1].backward(inp[0].data.T @ grad)

def pow_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for elementwise inp[0] ** inp[1] 
    """    
    assert len(inp) == 2


    inp[0].backward(grad * inp[1].data * (inp[0].data ** (inp[1].data - 1)))
    inp[1].backward(grad * np.log(inp[0].data) * (inp[0].data ** inp[1].data))

def exp_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for elementwise exp(inp[0])
    """
    assert len(inp) == 1

    inp[0].backward(grad * out)

def log_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for elementwise log(inp[0])
    """
    assert len(inp) == 1

    inp[0].backward(grad * 1/inp[0].data)

def relu_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for relu(inp[0])
    """
    assert len(inp) == 1

    inp[0].backward(grad * (inp[0].data > 0).astype(float))

def rowwise_softmax_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for rowwise_softmax(inp[0])
    """
    assert len(inp) == 1

    dot = np.sum(grad * out, axis=1, keepdims=True)
    inp[0].backward(out * (grad - dot))

def rowwise_logsumexp_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
    """
    Backward for rowwise_logsumexp(inp[0])
    """
    assert len(inp) == 1

    softmax = np_safe_rowwise_softmax(inp[0].data)
    inp[0].backward(grad * softmax)


def get_getitem_backward(idx: list[int | type(Ellipsis) | slice | None]) -> Callable[[np.ndarray, list["Tensor"], np.ndarray], None]: 
    """
    Create a backward function for a getitem operation.
    """
    def getitem_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
        assert len(inp) == 1

        new_grad = np.zeros_like(inp[0].data)
        new_grad[*idx] = grad

        inp[0].backward(new_grad)

    return getitem_backward


def get_sum_backward(axis: Optional[int]) -> Callable[[np.ndarray, list["Tensor"], np.ndarray], None]:
    """
    Create a backward function for a sum over this axis.
    """
    def sum_backward(grad: np.ndarray, inp: list["Tensor"], out: np.ndarray):
        assert len(inp) == 1

        if axis is not None:
            back_grad = np.expand_dims(grad, axis).repeat(inp[0].data.shape[axis], axis)
        else:
            back_grad = np.ones_like(inp[0].data) * grad
        
        inp[0].backward(back_grad)

    return sum_backward

class Tensor:
    def __init__(self, data: np.ndarray, grad: Optional[np.ndarray] = None, inp: Optional[list["Tensor"]] = None, backward_operation: Optional[Callable[[np.ndarray, list["Tensor"], np.ndarray], None]] = None, tag: Optional[str]=None):
        self.data = data
        self.grad = grad
        self.inp = inp
        self.backward_operation = backward_operation
        if tag:
            self.tag = tag
        else:
            self.tag = str(uuid4())[:6]
    
    def backward(self, grad: np.ndarray):
        if np.isnan(grad).any() or (np.abs(grad) > 1e10).any():
            breakpoint()
        added_grad = sum_broadcasted_axes(self.shape, grad)

        if self.grad is None:
            self.grad = added_grad
        else:
            self.grad += added_grad

        assert self.grad.shape == self.shape
        
        if self.inp is not None and self.backward_operation is not None:
            self.backward_operation(added_grad, self.inp, self.data)

    def zero_grad(self):
        self.grad = None
        self.inp = None
        self.backward_operation = None

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    def T(self) -> "Tensor":
        # self.T
        data = self.data.T
        return Tensor(data, inp=[self], backward_operation=transpose_backward, tag=f"({self.tag}).T")

    def __neg__(self) -> "Tensor":
        # -self
        data = -self.data
        return Tensor(data, inp=[self], backward_operation=neg_backward, tag=f"-{self.tag}")

    def __add__(self, o: "Tensor") -> "Tensor":
        # self + o
        data = self.data + o.data
        return Tensor(data, inp=[self, o], backward_operation=add_backward, tag=f"{self.tag} + {o.tag}")

    def __sub__(self, o: "Tensor") -> "Tensor":
        # self - o
        return self + (-o)

    def __mul__(self, o: "Tensor") -> "Tensor":
        # self * o
        data = self.data * o.data
        return Tensor(data, inp=[self, o], backward_operation=mul_backward, tag=f"({self.tag}) * ({o.tag})")

    def __matmul__(self, o: "Tensor") -> "Tensor":
        # self @ o
        data = self.data @ o.data
        return Tensor(data, inp=[self, o], backward_operation=matmul_backward, tag=f"({self.tag}) @ ({o.tag})")

    def __pow__(self, o: "Tensor") -> "Tensor":
        # self**o
        data = self.data**o.data
        return Tensor(data, inp=[self, o], backward_operation=pow_backward, tag=f"({self.tag}) ** ({o.tag})")

    def __truediv__(self, o: "Tensor") -> "Tensor":
        # self / o
        return self * (o ** Tensor(np.array(-1), tag="-1"))

    def __getitem__(self, idx) -> "Tensor":
        # self[idx]
        data = self.data[idx]
        return Tensor(data, inp=[self], backward_operation=get_getitem_backward(idx), tag=f"({self.tag})[{idx}]")

    def item(self):
        return self.data.item()

    def exp(self) -> "Tensor":
        # Elementwise exp(self)
        data = np.exp(self.data)
        return Tensor(data, inp=[self], backward_operation=exp_backward, tag=f"exp({self.tag})")

    def log(self) -> "Tensor":
        # Elementwise log(self)
        data = np.log(self.data)
        return Tensor(data, inp=[self], backward_operation=log_backward, tag=f"log({self.tag})")

    def sum(self, axis: Optional[int] = None) -> "Tensor":
        # self.sum(axis)
        data = self.data.sum(axis=axis)
        return Tensor(data, inp=[self], backward_operation=get_sum_backward(axis), tag=f"sum({self.tag})")

    def mean(self, axis: Optional[int] = None) -> "Tensor":
        # self.mean(axis)
        x = self.sum(axis)
        
        if axis is not None:
            denom = np.prod([self.shape[i] for i in axis])
        else:
            denom = np.prod(self.shape)

        x = x / Tensor(np.array(denom, dtype=float), tag=str(denom))
        return x

class LRSchedule:
    def next(self, step: int) -> float:
        raise NotImplementedError()


class ConstantLRSchedule:
    def __init__(self, lr: float):
        self.lr = lr

    def next(self, step_count: int) -> float:
        return self.lr


class ExponentialLRSchedule(LRSchedule):
    def __init__(self, lr: float, decay: float):
        assert 0 <= decay <= 1
        self.lr = lr
        self.decay = decay

    def next(self, step_count: int) -> float:
        self.lr = self.lr * self.decay

        return self.lr   


class Optimizer:
    def __init__(self, parameters: list[Tensor], lr_schedule: LRSchedule):
        self.parameters = parameters
        self.lr_schedule = lr_schedule
        self.step_count = 0

    def step(self): 
        raise NotImplementedError()

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        self.step_count += 1


class SGD(Optimizer):
    def step(self):
        super().step()

        lr = self.lr_schedule.next(self.step_count)

        # breakpoint()
        for param in self.parameters:
            param.data -= lr * param.grad


class Adam(Optimizer):
    def __init__(self, parameters: list[Tensor], lr_schedule: LRSchedule, beta1: float=0.9, beta2: float=0.999):
        super().__init__(parameters, lr_schedule)

        self.beta1 = beta1
        self.beta2 = beta2

        self.m1 = []
        self.m2 = []

        for param in parameters:
            self.m1.append(np.zeros_like(param.data))
            self.m2.append(np.zeros_like(param.data))

    def step(self):
        super().step()

        lr = self.lr_schedule.next(self.step_count)

        for i, param in enumerate(self.parameters):
            cur_m1 = param.grad
            cur_m2 = param.grad ** 2

            self.m1[i] = self.beta1 * self.m1[i] + (1 - self.beta1) * cur_m1
            self.m2[i] = self.beta2 * self.m2[i] + (1 - self.beta2) * cur_m2

            grad = (1 - self.beta2**self.step_count)**(1/2) * self.m1[i] / (np.sqrt(self.m2[i]) * (1 - self.beta1**self.step_count) + 1e-8)
            param.data -= lr * grad

def relu(x: Tensor) -> Tensor:
    data = x.data.copy()
    data[data < 0] = 0

    return Tensor(data, inp=[x], backward_operation=relu_backward, tag=f"relu({x.tag})")

def np_safe_rowwise_softmax(x: np.ndarray) -> np.ndarray:
    # Softmax is shift-invariant so we can subtract the row max
    x_shift = x - x.max(axis=1)[:, None]
    x_shift = np.exp(x_shift)
    return x_shift / x_shift.sum(axis=1)[:, None]

def rowwise_softmax(x: Tensor) -> Tensor:
    data = np_safe_rowwise_softmax(x.data)
    return Tensor(data, inp=[x], backward_operation=rowwise_softmax_backward, tag=f"softmax({x.tag})")

def rowwise_logsumexp(x: Tensor) -> Tensor:
    m = x.data.max(axis=1, keepdims=True)
    data = m + np.log(np.exp(x.data - m).sum(axis=1)[:, None])
    return Tensor(data, inp=[x], backward_operation=rowwise_logsumexp_backward, tag=f"logsumexp({x.tag})")

def rowwise_logsoftmax(x: Tensor) -> Tensor:
    return x - rowwise_logsumexp(x)
