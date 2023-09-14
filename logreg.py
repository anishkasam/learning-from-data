import numpy as np
import random

class LogisticRegression:
    def __init__(self, train_data, test_data):
        self.weights = np.array([0.0, 0.0, 0.0])
        self.train_data = train_data
        self.test_data = test_data

        # points for the target function's line
        self.x1, self.y1 = random.uniform(-1, 1), random.uniform(-1, 1)
        self.x2, self.y2 = random.uniform(-1, 1), random.uniform(-1, 1)
        self.slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        
    def target(self, point):
        # checks if the given coordinate is above or below the target function line
        return 1 if point[2] > (self.slope * (point[1] - self.x1) + self.y1) else -1
    
    # error function
    def cross_entropy(self, point, target):
        return np.log(1 + np.exp(-target * np.dot(self.weights, point)))
    
    # gradient function
    def gradient(self, point, target):
        return np.divide(-target * point, 1 + (np.exp(target * np.dot(self.weights, point))))

    def train(self):
        x_matrix = np.array(self.train_data)
        y_matrix = np.array([self.target(point) for point in self.train_data])

        lr = 0.01
        prev_weights = None
        epochs = 0
        # distance between the two weight vectors
        difference = 1

        while difference > 0.01:
            # randomly arange the indexes of the points for sgd
            indices = list(range(len(y_matrix)))
            random.shuffle(indices)
            prev_weights = self.weights

            for index in indices:
                # randomly selected point and target
                point = x_matrix[index]
                target = y_matrix[index]

                descent = self.gradient(point, target)
                self.weights = self.weights - lr * descent

            epochs += 1
            # || w^(t-1) - w^t ||
            difference = np.linalg.norm(prev_weights - self.weights)  
        
        return epochs

    def test(self):
        total_error = 0

        for point in self.train_data:
            target = self.target(point)
            total_error += self.cross_entropy(point, target)

        return total_error / len(self.train_data)

def lr_experiment(train_points, test_points, trials):
    epochs = 0
    error = 0

    for trial in range(trials):
        # generate training and testing datasets
        train = [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(train_points)]
        test = [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(test_points)]

        lr = LogisticRegression(train, test)

        epochs += lr.train()
        error += lr.test()

    avg_epochs = epochs / trials
    avg_error = error / trials
    
    return avg_epochs, avg_error

epochs, error = lr_experiment(100, 1000, 10)
print(epochs, error)
# 335.38, 0.09478627989385599