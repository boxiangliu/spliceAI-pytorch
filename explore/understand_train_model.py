from engine.trainer import Trainer
from utils import load_config
import numpy as np
from sklearn.metrics import average_precision_score

cfg_file = "config.yaml"
cfg = load_config(cfg_file)

trainer = Trainer(cfg_file)
seqs, labels = next(trainer.train_iter)
Xc = seqs.numpy().transpose((0,2,1))
Yc = labels.numpy().transpose((0,2,1))

is_expr = (Yc.sum(axis=(1,2)) >= 1)
Y_true = Yc[:, :, 1].flatten()


def print_topl_statistics(y_true, y_pred):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.
    breakpoint()
    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:

        idx_pred = argsorted_y_pred[-int(top_length * len(idx_true)):]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred))
                           / float(min(len(idx_pred), len(idx_true)))]
        threshold += [sorted_y_pred[-int(top_length * len(idx_true))]]

    auprc = average_precision_score(y_true, y_pred)

    print(("%.4f\t\033[91m%.4f\t\033[0m%.4f\t%.4f\t\033[94m%.4f\t\033[0m"
          + "%.4f\t%.4f\t%.4f\t%.4f\t%d") % (
        topkl_accuracy[0], topkl_accuracy[1], topkl_accuracy[2],
        topkl_accuracy[3], auprc, threshold[0], threshold[1],
        threshold[2], threshold[3], len(idx_true)))


print_topl_statistics(np.asarray(Y_true),
                      np.asarray(Y_true))
