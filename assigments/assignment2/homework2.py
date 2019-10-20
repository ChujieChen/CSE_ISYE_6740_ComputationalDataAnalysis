import scipy.io
from AccMeasure import acc_measure
from mycluster import cluster
from mycluster_extra import cluster_extra
from show_topics import display_topics

mat = scipy.io.loadmat('data.mat')
mat = mat['X']
X = mat[:, :-1]

idx = cluster(X, 4)

acc = acc_measure(idx)

print('accuracy %.4f' % (acc))

# ======================== uncomment the following for extra task ========================
# n_topics = None # TODO specify num topics yourself
# cell = scipy.io.loadmat('nips.mat')
# mat = cell['raw_count'] # sparse mat of size (num_doc, num_words)
# wl = cell['wl']
#
# W = cluster_extra(mat, n_topics)
#
# display_topics(W, wl)