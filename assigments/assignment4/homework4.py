import scipy.io
from algorithm import algo

mat = scipy.io.loadmat('sp500.mat')
mat = mat['price_move']


for q in [0.7, 0.9]:
    p, fig = algo(q, mat)

    fig.savefig('./'+str(q)+'.png')
    print('p: %.4f q: %.4f' % (p, q))