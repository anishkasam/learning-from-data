import pandas as pd
import numpy as np

# read in data files as csv and convert to pd dataframe
df_train = pd.read_csv('regularization/in.txt', names = ["x1", "x2", "y"], sep='\s+', dtype=np.float64)[25:]
df_val = pd.read_csv('regularization/in.txt', names = ["x1", "x2", "y"], sep='\s+', dtype=np.float64)[:25]
df_test = pd.read_csv('regularization/out.txt', names = ["x1", "x2", "y"], sep='\s+', dtype=np.float64)

# isolate columns as np arrays and linearly transform training and testing sets
def get_data(df):
    x1, x2, y = np.array(df['x1']), np.array(df['x2']), np.array(df['y'])
    z = np.array([np.ones(df.shape[0]), x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]).T
    return x1, x2, y, z

train_x1, train_x2, train_y, train_z = get_data(df_train)
val_x1, val_x2, val_y, val_z = get_data(df_val)
test_x1, test_x2, test_y, test_z = get_data(df_test)

def linreg(k):
    pinv = np.linalg.pinv(train_z[:, :(k + 1)])
    w = np.dot(pinv, train_y)

    in_error = sum(train_y != np.sign(np.dot(train_z[:, :(k + 1)], w))) / len(df_train)
    val_error = sum(val_y != np.sign(np.dot(val_z[:, :(k + 1)], w))) / len(df_val)
    out_error = sum(test_y != np.sign(np.dot(test_z[:, :(k + 1)], w))) / len(df_test)

    return (in_error, val_error, out_error)

for k in range(3, 8):
    error = linreg(k)
    in_error = error[0]
    val_error = error[1]
    out_error = error[2]
    print("k = {k}, E-in = {in_e}, E-val = {val_e}, and E-out = {out_e}".format(k = k, in_e = round(in_error, 4), val_e = round(val_error, 4), out_e = round(out_error, 4)))

# 1 & 2
'''
k = 3, E-in = 0.44, E-val = 0.3, and E-out = 0.42
k = 4, E-in = 0.32, E-val = 0.5, and E-out = 0.416
k = 5, E-in = 0.08, E-val = 0.2, and E-out = 0.188
k = 6, E-in = 0.04, E-val = 0.0, and E-out = 0.084
k = 7, E-in = 0.04, E-val = 0.1, and E-out = 0.072
'''

# 3 & 4
'''
k = 3, E-in = 0.4, E-val = 0.28, and E-out = 0.396
k = 4, E-in = 0.3, E-val = 0.36, and E-out = 0.388
k = 5, E-in = 0.2, E-val = 0.2, and E-out = 0.284
k = 6, E-in = 0.0, E-val = 0.08, and E-out = 0.192
k = 7, E-in = 0.0, E-val = 0.12, and E-out = 0.196
'''

# 5
# 0.072 and 0.192 are the answers from 1 & 3