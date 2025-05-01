import sys
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random

random.seed(0)
np.random.seed(0)


class GMM:
    """Gaussian Mixture Model

    Parameters
    -----------
        n_clusters: int , number of gaussian distributions
        n_runs: int, number of iterations to run algorithm, default: 200

    """

    def __init__(self, n_clusters, n_runs=200):
        self.C = n_clusters  # number of Guassians/clusters
        self.n_runs = n_runs

    def fit(self, X):
        """Compute the E-step and M-step and calculates the lowerbound

        Parameters:
        -----------
        X: (n_samples, n_features)

        Returns:
        ----------
        instance of GMM

        """

        self.mu, self.cov, self.pi = self.init_parameters(X)
        lower_bound = -np.inf

        try:
            for run in range(self.n_runs):
                prev_lower_bound = lower_bound
                self.e_step(X)
                self.m_step(X)
                lower_bound = self.compute_loss_function(X, self.pi, self.mu, self.cov)
                if abs(prev_lower_bound - lower_bound) < 1e-6:
                    break

        except Exception as e:
            print(e)

        return self

    def init_parameters(self, X):
        """Implement k-means to find starting parameter values.

        Parameters:
        ------------
        X: numpy array of data points, (n_samples, n_features)

        Returns:
        ----------
        initial_means: numpy array: (n_clusters, n_features)
        initial_cov: numpy array: (n_clusters, n_features, n_features)
        initial_pi: numpy array: (n_clusters)

        """

        n_samples, n_features = X.shape

        n_clusters = self.C

        # kmeans predict
        kmeans = KMeans(n_clusters=n_clusters, max_iter=500)
        kmeans.fit(X)
        pred = kmeans.predict(X)

        init_means = np.zeros((n_clusters, n_features))
        init_cov = np.zeros((n_clusters, n_features, n_features))
        init_pi = np.zeros(n_clusters)

        labels = np.unique(pred)

        group = 0

        for label in labels:
            ids = np.where(pred == label)
            init_pi[group] = len(ids[0]) / n_samples
            init_means[group, :] = np.mean(X[ids], axis=0)
            diff_mean = X[ids] - init_means[group, :]
            init_cov[group, :, :] = np.dot(diff_mean.T, diff_mean) / len(ids[0])
            group += 1

        return init_means, init_cov, init_pi

    def e_step(self, X):
        """
        Parameters
        ----------
        X: (n_samples, n_features)

        """

        n_samples, n_features = X.shape

        self.gamma = np.empty((n_samples, self.C))

        """
        TODO: assign the ownership weights of each data points
        
        self.gamma: the ownership weights of each data points, (n_samples, n_clusters)
        """
        for c in range(self.C):  # 對每個高斯分佈
            # 計算第 c 個高斯分佈的概率密度函數
            pd = multivariate_normal(self.mu[c], self.cov[c])
            self.gamma[:, c] = self.pi[c] * pd.pdf(X)  # 混合權重 * 概率密度值

        # 將每行歸一化，使得對應到所有高斯分佈的責任值和為 1
        self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)

    def m_step(self, X):
        """
        Parameters
        ----------
        X: (n_samples, n_features)

        """

        n_samples, n_features = X.shape
        n_clusters = self.C

        """
        TODO: calculate parameters(mean, covariance, mixing probability) by MLE
        
        self.mu: mean, (n_clusters, n_features)
        
        self.cov: covariance, (n_clusteres, n_features, n_features)
        
        self.pi: mixing probability, (n_clusters)
        """
        N_k = np.sum(self.gamma, axis=0)  # 每個高斯分佈的加權樣本數

        # 更新均值 mu
        self.mu = np.dot(self.gamma.T, X) / N_k[:, np.newaxis]

        # 更新協方差 cov
        self.cov = np.zeros((n_clusters, n_features, n_features))
        for c in range(n_clusters):
            diff = X - self.mu[c]  # 每個樣本與當前分佈均值的偏差
            self.cov[c] = np.dot(self.gamma[:, c] * diff.T, diff) / N_k[c]

        # 更新混合比例 pi
        self.pi = N_k / n_samples

    def compute_loss_function(self, X, pi, mu, cov):
        """Computes lower bound loss function

        Parameters:
        -----------
        X: (n_samples, n_features)
        pi: mixing probability, (n_clusters)
        mu: mean, (n_clusters, n_features)
        cov: covariance, (n_clusteres, n_features, n_features)

        Returns:
        ---------
        loss: float
        """

        n_samples, n_features = X.shape
        n_clusters = self.C
        loss = np.empty((n_samples, n_clusters))

        for c in range(n_clusters):
            pd = multivariate_normal(self.mu[c], self.cov[c])
            loss[:, c] = self.gamma[:, c] * (
                np.log(self.pi[c] + 1e-6)
                + pd.logpdf(X)
                - np.log(self.gamma[:, c] + 1e-6)
            )

        loss = np.sum(loss)

        return loss

    def predict(self, X):
        """
        Parameters:
        -------------
        X: (n_samples, n_features)

        Returns:
        ----------
        labels: predicted cluster based on highest gamma, (n_samples)

        """
        n_samples, n_features = X.shape
        labels = np.zeros((n_samples, self.C))

        for c in range(self.C):
            labels[:, c] = self.pi[c] * multivariate_normal.pdf(
                X, self.mu[c, :], self.cov[c]
            )

        labels = labels.argmax(1)

        return labels


def infer_cluster_labels(pred_labels, actual_labels):
    """
    Associates most probable label with each cluster in gmm model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}
    n_clusters = 3
    for i in range(n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(pred_labels == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """

    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels


def main(argv):
    iris = datasets.load_iris()
    gmm_train_acc = []
    gmm_test_acc = []

    for run in range(10):
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3
        )
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)

        gmm = GMM(n_clusters=3, n_runs=100)
        fitted_values = gmm.fit(x_train)

        # train predict
        train_predict_y = gmm.predict(x_train)
        cluster_labels = infer_cluster_labels(train_predict_y, y_train)
        train_predicted_labels = infer_data_labels(train_predict_y, cluster_labels)
        gmm_train_acc.append(
            np.count_nonzero(y_train == train_predicted_labels) / len(y_train)
        )

        # test predict
        test_predict_y = gmm.predict(x_test)
        cluster_labels = infer_cluster_labels(test_predict_y, y_test)
        test_predicted_labels = infer_data_labels(test_predict_y, cluster_labels)
        gmm_test_acc.append(
            np.count_nonzero(y_test == test_predicted_labels) / len(y_test)
        )

    print("gmm train acc: ", np.mean(gmm_train_acc), "+-", np.std(gmm_train_acc))
    print("gmm test acc: ", np.mean(gmm_test_acc), "+-", np.std(gmm_test_acc))


if __name__ == "__main__":
    main(sys.argv)
