from typing import Union
import numpy as np

from tools import load_iris, split_train_test
import matplotlib.pyplot as plt

def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''

    x = np.clip(x, -100, None)  # Will return 0.0 in case of overflow
    sigm = (1 / (1 + np.exp(-x)))
    
    return sigm


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x) * (1 - sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    activation_fct = 0
    sum = 0
    for i in range(len(x)):
        sum += x[i] * w[i]
    activation_fct = sigmoid(sum)

    return sum, activation_fct
        


def ffnn(
    x: np.ndarray,
    M: int,     # nb of hidden layer neurons
    K: int,     # nb of output neurons
    W1: np.ndarray, # linear transform from input to hidden layer
    W2: np.ndarray, # linear transform from hidden layer to output
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x, 0, 1.0)
    a1, z1 = [], []
    for i in range(M):
        weighted_sum, output = perceptron(z0, W1[:, i])
        a1.append(weighted_sum)
        z1.append(output)
    
    z1 = np.insert(z1, 0, 1.0)
    a2, y = [], []

    for i in range(K):
        weighted_sum, output = perceptron(z1, W2[:, i])
        a2.append(weighted_sum)
        y.append(output)

    return np.array(y), np.array(z0), np.array(z1), np.array(a1), np.array(a2)


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    delta_k = y - target_y
    delta_j = d_sigmoid(a1) * np.dot(W2[1:, :], delta_k)

    dE1 = np.zeros(np.shape(W1))
    dE2 = np.zeros(np.shape(W2))

    dE1 += np.outer(z0, delta_j)
    dE2 += np.outer(z1, delta_k)

    return y, dE1, dE2

def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    E_total = []
    misclassification_rate = []

    for _ in range(iterations):
        dE1_total = np.zeros_like(W1)
        dE2_total = np.zeros_like(W2)
        misclassifications = 0
        E = 0

        for i in range(len(X_train)):
            x = X_train[i, :]
            t = t_train[i]
            target_y = np.zeros(K)
            target_y[t] = 1.0


            # BackPropagation: get gradient matrices and output values
            y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

            dE1_total += dE1
            dE2_total += dE2

            if np.argmax(y) != t_train[i]:
                misclassifications += 1     # if misclassified, +1
            
            E -= np.sum(target_y * np.log(y) + (1 - target_y) * np.log(1 - y))
            

        # We adjust the weight
        N = len(X_train)
        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N


        E_total.append(E / N)
        misclassification_rate.append(misclassifications / N)

    last_guesses = []

    for j in range(N):
        y, z0, z1, a1, a2 = ffnn(X_train[j, :], M, K, W1, W2)
        last_guesses.append(np.argmax(y))

    return W1, W2, E_total, misclassification_rate, last_guesses




def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = len(X)
    guesses = np.zeros(N, dtype= int)

    for i in range (N):
        y, z0, z1, a1, a2 = ffnn(X[i, :], M, K, W1, W2)
        output = np.argmax(y)
        guesses[i] = output

    return guesses




def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    
    length_predictions = len(prediction)
    matrix = np.zeros((length_predictions, length_predictions), int)

    for i in range(len(target)):
        current_class = target[i]
        predicted_class = prediction[i]
        matrix[predicted_class][current_class] += 1
        
    return matrix

def accuracy(
        prediction: np.ndarray, 
        target: np.ndarray
        ) -> np.ndarray:
    correct_predictions = np.sum(prediction == target)    
    accuracy = 100*correct_predictions / len(prediction)
    return accuracy




if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """

    
    #1.1
    print(sigmoid(0.5))
    print(d_sigmoid(0.2))
    #1.2
    print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))
    

    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
    # initialize the random generator to get repeatable results
    np.random.seed(1234)
    # Take one point:
    x = train_features[0, :]
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    
    #1.3
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    
    print("y:", y)
    print("z0:", z0)
    print("z1:", z1)
    print("a1:", a1)
    print("a2:", a2)
    

    #1.4
    # initialize random generator to get predictable results
    np.random.seed(42)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    x = features[0, :]

    # create one-hot target for the feature
    target_y = np.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    
    print("y: ", y)
    print("dE1: \n", dE1)
    print('dE2: \n', dE2)
    


    #2.1

    # initialize the random seed to get predictable results
    np.random.seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    
    print("W1tr: \n", W1tr)
    print("W2tr: \n", W2tr)
    print("Etotal: \n", Etotal)
    print("Missclassification rate: \n", misclassification_rate)
    print("Last guesses: \n", last_guesses)
    

    #2.2
    #print(test_nn(test_features, M, K, W1tr, W2tr))

    #2.3
    
    #Training
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:, :], train_targets[:], M, K, W1, W2, 500, 0.1)
    
    print("Etotal: \n", Etotal)
    print("Missclassification rate: \n", misclassification_rate)

    #Testing
    test_pred = test_nn(test_features, M, K, W1tr, W2tr)
    
    print("Test on dataset: \n", test_pred)
    print("Test Targets: \n", test_targets)
    print("Accuracy: \n", accuracy(test_pred, test_targets))
    #Confusion matrix
    print("Confusion Matrix: \n", confusion_matrix(test_pred, test_targets))
    
    plt.subplot(121)
    plt.plot(Etotal)
    plt.xlabel("Iterations")
    plt.ylabel("Total Error")

    plt.subplot(122)
    plt.plot(misclassification_rate)
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification Rate")
    plt.tight_layout()
    plt.show()
    
    
    