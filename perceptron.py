import numpy as np
import random

class Perceptron:
    def __init__(self, train_data, test_data):
        # weights has length of 3 for threshold, x, and y
        self.weights = np.array([0.0, 0.0, 0.0])
        self.train_data = train_data
        self.test_data = test_data

        # points for the target function's line
        self.x1, self.y1 = random.uniform(-1, 1), random.uniform(-1, 1)
        self.x2, self.y2 = random.uniform(-1, 1), random.uniform(-1, 1)
        self.slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        
    def target(self, point):
        x = point[1]
        y = point[2]
        # checks if the given coordinate is above or below the target function line
        return np.sign(y - (self.slope * (x - self.x1) + self.y1))
    
    def hypothesis(self, point):
        # checks if dot of x and y with the weights is greater than the threshold
        return np.sign(np.dot(self.weights, np.array(point)))
    
    def train(self):
        iterations = 0
        
        while True:
            misclassified = []
            # iterate through all of the training points
            for point in self.train_data:
                if self.hypothesis(point) != self.target(point):
                    misclassified.append([point[1], point[2], self.target(point)])

            # check if all the points were classified correctly
            if len(misclassified) == 0:
                break
            else:
                iterations += 1
                # randomly select one of the misclassified points and adjust the weight vector
                misclass = random.choice(misclassified)
                update = np.array([1, misclass[0], misclass[1]]) * misclass[2]
                self.weights += update

        return iterations
    
    def test(self):
        disagreements = 0

        for point in self.test_data:
            if self.hypothesis(point) != self.target(point):
                disagreements += 1

        return disagreements / len(self.test_data) 

def experiment(points, trials):
    total_iterations = 0
    total_error = 0

    for trial in range(trials):
        # generate training and testing datasets
        train = [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(points)]
        test = [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(points)]

        pla = Perceptron(train, test)

        total_iterations += pla.train()
        total_error += pla.test()

    avg_iterations = total_iterations / trials
    avg_error = total_error / trials
    
    return avg_iterations, avg_error

iterations, error = experiment(10, 1000)
print(iterations, error)
# 9.139, 0.105

iterations, error = experiment(100, 1000)
print(iterations, error)
# 105.883, 0.014
