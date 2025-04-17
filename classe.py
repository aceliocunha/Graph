import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    
    def sigmoid_derivative(self, y):
        return y * (1 - y)
    
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1) 
        for _ in range(self.epochs):
            z1 = np.dot(X, self.weights1) + self.bias1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.weights2) + self.bias2
            a2 = self.sigmoid(z2)
            
            loss = y - a2
            
            d_a2 = loss * self.sigmoid_derivative(a2)  
            d_weights2 = np.dot(a1.T, d_a2)  
            d_bias2 = np.sum(d_a2, axis=0, keepdims=True)  
            
            d_a1 = np.dot(d_a2, self.weights2.T) * self.sigmoid_derivative(a1)  
            d_weights1 = np.dot(X.T, d_a1) 
            d_bias1 = np.sum(d_a1, axis=0, keepdims=True)  
            
            self.weights2 += self.learning_rate*d_weights2
            self.bias2 += self.learning_rate*d_bias2
            self.weights1 += self.learning_rate*d_weights1
            self.bias1 += self.learning_rate*d_bias1

    
    def predict(self, X):
        z_1 = np.dot(X, self.weights1) + self.bias1
        a1 = self.sigmoid(z_1)
        z_2 = np.dot(a1, self.weights2) + self.bias2
        return (self.sigmoid(z_2) > 0.5).astype(int)