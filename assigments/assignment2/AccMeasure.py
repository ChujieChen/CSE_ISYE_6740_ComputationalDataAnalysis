import scipy.io
import itertools
import numpy as np


def acc_measure(idx):
    """

    :param idx:
        numpy array of (num_doc)
    :return:
        accuracy
    """

    mat = scipy.io.loadmat('data.mat')
    mat = mat['X']
    Y = mat[:, -1]

    # rotate for different idx assignments
    best_acc = 0
    for idx_order in itertools.permutations([1, 2, 3, 4]):

        for ind, label in enumerate(idx_order):
            Y[(ind)*100:(ind+1)*100] = label

        acc = (Y == idx).sum() / Y.shape[0]
        if acc > best_acc:
            best_acc = acc

    return best_acc


if __name__ == '__main__':
    acc_measure(np.array([1]*100 + [3]*100 + [2]*100 + [4]*100))
