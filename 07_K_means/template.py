# Author: Yann DURIEZ
# Date: 07/10/2023
# Project: K-means Clustering
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    n = X.shape[0]
    k = Mu.shape[0]
    eucl_dist = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            eucl_dist[i, j] = np.sqrt(np.sum(np.square(X[i] - Mu[j])))

    return eucl_dist


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    n, k = dist.shape
    r_out = np.zeros((n, k))

    for i in range(n):
        index_closest = np.argmin(dist[i])
        r_out[i, index_closest] = 1

    return r_out


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    n, k = R.shape
    J = 0.0

    for i in range(n):
        for j in range(k):
            J += R[i, j] * dist[i, j]

    return J / n

def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''

    denominator = np.sum(R, axis=0)
    updated_Mu = np.dot(R.T, X) / denominator[:, np.newaxis]

    return updated_Mu


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    
    '''
    You should run the algorithm num_its times. 
    For each iteration we collect the value of the objective function, J(hat).

    k_means should return:

    Mu: The last values of the prototypes
    R: The last value of the indicators
    Js: List of J(hat) values for each iteration of the algorithm.
    '''

    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.
    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    Js = []

    for _ in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        J_hat = determine_j(R, dist)
        Mu = update_Mu(Mu, X_standard, R)

        Js.append(J_hat)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js


def _plot_j(Js):
    plt.plot(Js)
    plt.show()


def _plot_multi_j(
    X: np.ndarray,
    k: np.ndarray,
    num_its: int):
    
    for i in range(len(k)):
        Mu, R, Js = k_means(X, k[i], num_its)
        plt.subplot(2, 2, i + 1)
        plt.plot(Js)
        plt.title(f'K = {k[i]}')
        plt.ylabel('Js')
    plt.tight_layout()
    plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    Mu, R, Js = k_means(X, len(classes), num_its)

    cluster_target = np.argmax(R, axis=1)

    cluster_class = {}
    for class_label in classes:
        cluster_cnt = np.bincount(cluster_target[t == class_label])
        mst_common_cluster = np.argmax(cluster_cnt)
        cluster_class[mst_common_cluster] = class_label

    prediction, k_means_predictions = [], []
    for cluster in cluster_target:
        prediction = cluster_class.get(cluster, None)
        if prediction is None:
            prediction=0
        
        k_means_predictions.append(prediction)
        
    return np.array(k_means_predictions, dtype=float)


def _iris_kmeans_accuracy(
    prediction: np.ndarray, 
    target: np.ndarray
) -> np.ndarray:
    
    #ACCURACY
    correct_predictions = np.sum(prediction == target)    
    accuracy = 100*correct_predictions / len(prediction)
    print(f'Accuracy:\n{accuracy}')

    #CONFUSION MATRIX
    length_predictions = len(prediction)
    matrix = np.zeros((length_predictions, length_predictions), int)

    for i in range(len(target)):
        current_class = target[i]
        predicted_class = prediction[i]
        matrix[predicted_class][current_class] += 1
        
    print(f'Confusion matrix:\n {matrix}')  



def _my_kmeans_on_image():
    pass


def plot_image_clusters(n_clusters: int):
    
    image, (w, h) = image_to_numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5, max_iter=100).fit(image)
    #kmeans = kmeans_prediction.fit(image)

    # Plot the clusters f
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


'''
## TO REMOVE BEFORE SENDING
#1.1
a = np.array([
    [1, 0, 0],
    [4, 4, 4],
    [2, 2, 2]])
b = np.array([
    [0, 0, 0],
    [4, 4, 4]])

#print(distance_matrix(a, b))

#1.2
dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
R = determine_r(dist)
#print(R)

#1.3
#print(determine_j(R, dist))

#1.4
X = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]])
Mu = np.array([
    [0.0, 0.5, 0.1],
    [0.8, 0.2, 0.3]])
R = np.array([
    [1, 0],
    [0, 1],
    [1, 0]])
#print(update_Mu(Mu, X, R))

#1.5
X, y, c = load_iris()
#print(k_means(X, 4, 10))

#1.6
Mu, R, Js = k_means(X, 4, 10)
#_plot_j(Js)

k = [2, 3, 5, 10]

#1.7
#_plot_multi_j(X, k, 10)

#1.9
print(k_means_predict(X, y, c, 5))

#1.10
#_iris_kmeans_accuracy(k_means_predict(X, y, c, 5), y)

#2.1
image, (w, h) = image_to_numpy()
#print(k_means(image, 7, 5))

#kmeans = KMeans(n_clusters= 7, random_state=0, n_init=5, max_iter=100)
#print(kmeans.fit_predict(image))    #print cluster labels


num_clusters = [2, 5, 10, 20]

#for cluster in num_clusters:
#    plot_image_clusters(cluster)'''