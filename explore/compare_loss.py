import torch
import torch.nn.functional as F

y_true = torch.randint(low=0, high=3, size=(48, 5000))
y_true = F.one_hot(y_true, num_classes=3).transpose(2, 1).float()

y_pred = torch.randn(size=(48, 3, 5000))
y_norm = F.softmax(y_pred, dim=1)

F.cross_entropy(y_pred, y_true, reduction="none")
F.cross_entropy(y_pred, y_norm)


def categorical_crossentropy_2d(y_true, y_pred):
    # Standard categorical cross entropy for sequence outputs

    return - torch.mean(y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + 1e-10)
                        + y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + 1e-10)
                        + y_true[:, 2, :] * torch.log(y_pred[:, 2, :] + 1e-10))

categorical_crossentropy_2d(y_true, y_norm)

y_true[:, 0, :] * torch.log(y_norm[:, 0, :])