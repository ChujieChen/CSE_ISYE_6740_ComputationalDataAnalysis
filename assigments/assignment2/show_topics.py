import numpy as np
import scipy.io


def display_topics(W, wl):
    """

    :param W:
        word-topic matrix of size (V, K)
    :param wl:
        array of str of size (V)
    :return:
    """
    top_n_words = 6
    ind_mat = np.argsort(-W.T, axis=1)[:, :top_n_words]

    for k_ind in range(ind_mat.shape[0]):
        w_ls = [wl[ind, 0][0] for ind in ind_mat[k_ind]]
        print('topic %i: %s' % (k_ind, ' '.join(w_ls)))


if __name__ == '__main__':
    cell = scipy.io.loadmat('nips.mat')
    mat = cell['raw_count'] # sparse mat of size (num_doc, num_words)
    wl = cell['wl']
    display_topics(np.random.rand(5,5), wl)