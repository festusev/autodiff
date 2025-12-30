import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
import math
from autodiff import *

EPOCHS = 100
B = 128
LR = 1e-3
ds = load_dataset("mulsi/fruit-vegetable-concepts")


def label_to_map(col) -> dict[str, int]:
    values = set(col)
    mp = {key: i for i, key in enumerate(values)}
    return mp


features = [
    "sphere",
    "cube",
    "cylinder",
    "red",
    "green",
    "orange",
    "yellow",
    "stem",
    "leaf",
    "tail",
    "seed",
    "pulp",
    "soil",
    "tree",
    "ovaloid",
    "blue",
    "brown",
    "white",
    "black",
    "purple",
]


def featurize(*row):
    out = {}
    for i, feature in enumerate(features):
        # Temporary fix
        if row[i] is None:
            out[feature] = -1
        else:
            out[feature] = int(row[i])
    return out


# def rowwise_softmax(x: np.ndarray) -> np.ndarray:
#     # Softmax is shift-invariant so we can subtract the row max
#     x = x - x.max(axis=1)[:, None]
#     x = np.exp(x)
#     x = x / x.sum(axis=1)[:, None]
#     return x


def forward(
    x0: Tensor, weights: list[Tensor], bias: list[Tensor]
) -> tuple[Tensor, dict]:
    # Run a forward pass
    z1 = x0 @ weights[0] + bias[0]
    x1 = relu(z1)
    z2 = x1 @ weights[1] + bias[1]
    x2 = relu(z2)
    z3 = x2 @ weights[2] + bias[2]
    x3 = relu(z3)
    z4 = x3 @ weights[3] + bias[3]
    out = rowwise_logsoftmax(z4)
    
    return out, {
        "x0": x0.data,
        "z1": z1.data,
        "x1": x1.data,
        "z2": z2.data,
        "x2": x2.data,
        "z3": z3.data,
        "x3": np_safe_rowwise_softmax(z3.data),
    }


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


ds = ds.map(featurize, input_columns=features)

class_map = label_to_map(ds["train"]["class"])
ds = ds.map(
    lambda class_label: {"class": class_map[class_label]}, input_columns="class"
)

# Initialize 3 layer network
weights = [
    Tensor(np.random.normal(size=(len(features), 100))/np.sqrt(100), tag="w0"),
    Tensor(np.random.normal(size=(100, 100))/np.sqrt(100), tag="w1"),
    Tensor(np.random.normal(size=(100, 100))/np.sqrt(100), tag="w2"),
    Tensor(np.random.normal(size=(100, len(class_map)))/np.sqrt(len(class_map)), tag="w3"),
]
bias = [
    Tensor(np.random.normal(size=100)/np.sqrt(100), tag="b0"),
    Tensor(np.random.normal(size=100)/np.sqrt(100), tag="b1"),
    Tensor(np.random.normal(size=100)/np.sqrt(100), tag="b2"),
    Tensor(np.random.normal(size=len(class_map))/np.sqrt(len(class_map)), tag="b3"),
]


lr_schedule = ExponentialLRSchedule(lr=LR, decay=0.99)
optim = Adam(weights + bias, lr_schedule)

for epoch in range(EPOCHS):
    print(f"<< EPOCH {epoch} >>")

    ds["train"] = ds["train"].shuffle()

    batch_losses = []
    batch_accuracies = []
    pbar = tqdm(range(math.ceil(len(ds["train"]) / B)))
    for i in pbar:
        batch = ds["train"].select(range(i * B, min((i + 1) * B, len(ds["train"]))))

        x0 = Tensor(np.stack([batch[feat] for feat in features], dtype=float).T, tag="x0")
        y = Tensor(np.array(batch["class"]), tag="y")

        # Forward pass
        out, hidden = forward(x0, weights, bias)
        
        # To compute final predictions, take max at the end
        y_pred = out.data.argmax(axis=1)
        accuracy = y.data == y_pred
        batch_accuracies.append(accuracy)
        accuracy = accuracy.mean().item()

        # Compute Loss
        loss = -out[np.arange(len(batch)), y.data]
        batch_losses.append(loss.data)
        loss = loss.mean()

        # Backprop
        loss.backward(np.array(1.0))
        optim.step()

        # Double check backprop by calculating by hand
        dLdz3 = hidden["x3"]
        dLdz3[np.arange(len(batch)), y.data] -= 1
        dLdw3 = hidden["x2"].T @ dLdz3 / len(batch)
        dLdb3 = dLdz3.sum(axis=0) / len(batch)

        dLdz2 = dLdz3 @ weights[2].data.T * relu_derivative(hidden["z2"])
        dLdw2 = hidden["x1"].T @ dLdz2 / len(batch)
        dLdb2 = dLdz2.sum(axis=0) / len(batch)

        dLdz1 = dLdz2 @ weights[1].data.T * relu_derivative(hidden["z1"])
        dLdw1 = hidden["x0"].T @ dLdz1 / len(batch)
        dLdb1 = dLdz1.sum(axis=0) / len(batch)
        # # Update
        # weights[2] -= alpha * dLdw3
        # weights[1] -= alpha * dLdw2
        # weights[0] -= alpha * dLdw1

        # bias[2] -= alpha * dLdb3
        # bias[1] -= alpha * dLdb2
        # bias[0] -= alpha * dLdb1

        # Update pbar
        pbar.set_description(f"Loss: {loss.item():0.2f} Accuracy: {accuracy:0.2f}")

    batch_loss = np.concatenate(batch_losses).mean().item()
    batch_accuracy = np.concatenate(batch_accuracies).mean().item()

    print(f"Batch Loss: {batch_loss:0.2f} Batch Accuracy: {batch_accuracy:0.2f}")
breakpoint()
