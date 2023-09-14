import pandas as pd
import numpy as np

# read in data files as csv and convert to pd dataframe
df_train = pd.read_csv('regularization/in.txt', names = ["x1", "x2", "y"], sep='\s+', dtype=np.float64)
df_test = pd.read_csv('regularization/out.txt', names = ["x1", "x2", "y"], sep='\s+', dtype=np.float64)

# isolate columns as np arrays and linearly transform training and testing sets
train_x1, train_x2, train_y = np.array(df_train['x1']), np.array(df_train['x2']), np.array(df_train['y'])
train_z = np.array([np.ones(df_train.shape[0]), train_x1, train_x2, train_x1 ** 2, train_x2 ** 2, train_x1 * train_x2, abs(train_x1 - train_x2), abs(train_x1 + train_x2)]).T

test_x1, test_x2, test_y = np.array(df_test['x1']), np.array(df_test['x2']), np.array(df_test['y'])
test_z = np.array([np.ones(df_test.shape[0]), test_x1, test_x2, test_x1 ** 2, test_x2 ** 2, test_x1 * test_x2, abs(test_x1 - test_x2), abs(test_x1 + test_x2)]).T

def linreg(lam = None):
    if lam is None:
        pinv = np.linalg.pinv(train_z)
        w = np.dot(pinv, train_y)
    else:
        # compute w reg using given lambda value
        pinv = np.dot(np.linalg.inv(np.dot(train_z.T, train_z) + (10 ** lam) * np.identity(train_z.shape[1])), train_z.T)
        w = np.dot(pinv, train_y)

    in_error = sum(train_y != np.sign(np.dot(train_z, w))) / len(df_train)
    out_error = sum(test_y != np.sign(np.dot(test_z, w))) / len(df_test)

    return (in_error, out_error)

for k in range(-5, 5):
    error = linreg(k)
    in_error = error[0]
    out_error = error[1]
    print("k = {k}, E-in = {in_e}, and E-out = {out_e}".format(k = k, in_e = round(in_error, 4), out_e = round(out_error, 4)))

'''
k = -5, E-in = 0.0286, and E-out = 0.084
k = -4, E-in = 0.0286, and E-out = 0.084
k = -3, E-in = 0.0286, and E-out = 0.08
k = -2, E-in = 0.0286, and E-out = 0.084
k = -1, E-in = 0.0286, and E-out = 0.056
k = 0, E-in = 0.0, and E-out = 0.092
k = 1, E-in = 0.0571, and E-out = 0.124
k = 2, E-in = 0.2, and E-out = 0.228
k = 3, E-in = 0.3714, and E-out = 0.436
k = 4, E-in = 0.4286, and E-out = 0.452
'''