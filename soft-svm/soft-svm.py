import numpy as np
from sklearn import svm

train = np.loadtxt("soft-svm/features.train")
test = np.loadtxt("soft-svm/features.test")

def parse_data(n, type):
    if type == "train":
        data = train
    else:
        data = test
    
    is_n = data[:,0] == n
    is_not_n = data[:,0] != 0
    x = data.copy()
    x[:,0] = 1.0
    y = 2 * is_n - 1
    return x, y

# for n in (0, 2, 4, 6, 8):
#     clf = svm.SVC(C = 0.01, kernel = "poly", degree = 2, gamma = 1.0)
#     x, y = parse_data(n, "train")
#     clf.fit(x, y)
#     y_predict = clf.predict(x)
#     e_in = np.mean(y != y_predict)
#     print(n, e_in)
'''
0 0.10588396653408312
2 0.10026059525442327
4 0.08942531888629818
6 0.09107118365107666
8 0.07433822520916199
'''

# for n in (1, 3, 5, 7, 9):
#     clf = svm.SVC(C = 0.01, kernel = "poly", degree = 2, gamma = 1.0)
#     x, y = parse_data(n, "train")
#     clf.fit(x, y)
#     y_predict = clf.predict(x)
#     e_in = np.mean(y != y_predict)
#     print(n, e_in)
'''
1 0.014401316691811822
3 0.09024825126868742
5 0.07625840076807022
7 0.08846523110684405
9 0.08832807570977919
'''