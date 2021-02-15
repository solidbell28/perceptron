import numpy as np


class Perceptron:
    '''The class representing a perceptron model'''

    def __init__(self, eta=0.01, n=10):
        # n - amount of iterations
        if (eta <= 0) or (eta >= 1):
            raise ValueError("Eta must be greater than 0 and less than 1")
        self.eta = eta
        self.n = n

    def fit(self, X, y):
        """Model training function"""
        self.coefs = np.zeros(X.shape[1])
        self.interception = 0
        self.errors = []
        # We will use the 'errors' list in future for some data visualization
        for _ in range(self.n):
            errors = 0.0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.coefs += update * xi
                self.interception += update
                errors += int(update != 0.0)
            self.errors.append(errors)

    def predict(self, X):
        """Result prediction function"""
        result = np.dot(X, self.coefs) + self.interception
        if result < -2:
            return 0
        elif result > 2:
            return 2
        else:
            return 1

    def score(self, X, y):
        """The function of evaluating the result of work of the model"""
        results = np.array([self.predict(xi) for xi in X])
        return np.sum(results == y) / len(y)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
Y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)

for eta, n in [(eta, n) for eta in [.1, .3, .5, .7, .9] for n in (10, 15, 20)]:
    pr = Perceptron(eta=eta, n=n)
    pr.fit(X_train, y_train)
    score = pr.score(X_test, y_test)
    errors = pr.errors

    # Visualizing the difference of scores of the perceptron with different options
    plt.figure(f"Perceptron(eta={eta}, n={n}). Score: {score}.")

    # Visualizing the dependence of amount of the iterations on the amount of the errors
    plt.xlabel("Amount of iterations")
    plt.ylabel("Amount of errors")
    plt.xticks(range(1, n + 1), [str(i) for i in range(1, n + 1)])
    plt.xlim(1, n)
    plt.plot(range(1, n + 1), errors, color="red")
    plt.grid()
    plt.show()
