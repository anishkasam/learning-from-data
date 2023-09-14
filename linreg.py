import numpy as np
import random

class LinearRegression:
    def __init__(self, train_data, test_data):
        # weights has length of 3 for threshold, x, and y
        self.weights = np.array([0.0, 0.0, 0.0])
        self.train_data = train_data
        self.test_data = test_data
    
    def hypothesis(self, point):
        # checks if dot of x and y with the weights is greater than the threshold
        return np.sign(np.dot(self.weights, np.array(point)))
    
    def train(self):
        x_matrix = np.array(self.train_data)
        y_matrix = np.array([self.target(point) for point in self.train_data])

        # gets the pseudo inverse of the input (x) matrix
        pinv = np.linalg.pinv(x_matrix)
        self.weights = np.dot(pinv, y_matrix)

        misclassified = 0
        for point in self.train_data:
            if self.target(point) != self.hypothesis(point):
                misclassified += 1

        return misclassified / len(self.train_data)
    
    def test(self):
        disagreements = 0
        for point in self.test_data:
            if self.hypothesis(point) != self.target(point):
                disagreements += 1

        return disagreements / len(self.test_data) 

def lr_experiment(train_points, test_points, trials):
    in_sample = 0
    out_sample = 0

    for trial in range(trials):
        # generate training and testing datasets
        train = [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(train_points)]
        test = [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(test_points)]

        lr = LinearRegression(train, test)

        in_sample += lr.train()
        out_sample += lr.test()

    avg_in_error = in_sample / trials
    avg_out_error = out_sample / trials
    
    return avg_in_error, avg_out_error

in_error, out_error = lr_experiment(100, 1000, 1000)
print(in_error, out_error)
# 0.0408, 0.0483