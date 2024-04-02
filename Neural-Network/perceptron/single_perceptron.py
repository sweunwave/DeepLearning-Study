import random
import numpy as np
import typing

import time

from functions import transform_functions as tf

import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,
                 weight :np.ndarray=np.random.rand(2)*2-1,
                 bias :int=0,
                 learning_rate :float=1.0) -> None:
        
        self.bias = bias
        self.weight = weight
        self.learning_rate = learning_rate
        self.error = 0.0
        self.a = None

    def predict(self, p :np.ndarray, t: int) -> bool:
        self.t = t
        self.p = p
        self.a = tf.hardlim((self.weight @ self.p) + self.bias)
        is_valid = (self.a == self.t)
        return is_valid

    def update(self):
        self.error = self.t - self.a
        self.weight = self.weight + self.error*self.p
        self.bias = self.bias + self.error

def main():
    # sample data : Bitwise AND operation
    p1 = np.array([0, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 0])
    p4 = np.array([1, 1])
    
    t1 = 0
    t2 = 1
    t3 = 1
    t4 = 1

    P = np.array([p1, p2, p3, p4])
    T = np.array([t1, t2, t3, t4])
    result = [False, False, False, False]

    is_valid_all = False

    perceptron = Perceptron(bias=np.random.randn(1))
    while (is_valid_all == False):
        for i in range(len(P)):
            result[i] = perceptron.predict(P[i], T[i])
            perceptron.update()
        is_valid_all = True

        # 전부 유효한지 검사
        for i in range(len(P)):
            if result[i] == False:
                is_valid_all = False
                break
        
    print("got a classification")
    print(f"weight vector : {perceptron.weight}")
    print(f"bias : {perceptron.bias}")

    # Result Visualization
    x_values = np.linspace(-1, 2, 100)
    y_values = -(perceptron.weight[0] * x_values + perceptron.bias) / perceptron.weight[1]

    plt.plot(x_values, y_values, color='r')
    plt.axis('equal')
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.scatter(P[:, 0], P[:, 1], c=T)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()