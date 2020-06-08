import numpy as np
import matplotlib.pyplot as plt


# simple dataset
def createDataset_1():
    N = 200
    mu_1 = [6.5, 2]
    mu_2 = [1.5, 2]
    Sigma_1 = [[0.8, 0],
               [0, 0.7]]
    Sigma_2 = [[1, 0],
               [0, 0.2]]
    X = []
    t = []

    for i in range(0, int(N / 2)):
        X.append(np.random.multivariate_normal(mu_1, Sigma_1))
        t.append(1)

    for i in range(0, int(N / 2)):
        X.append(np.random.multivariate_normal(mu_2, Sigma_2))
        t.append(-1)

    return np.asarray(X), np.asarray(t).reshape(N, 1)


# half-moon dataset
def createDataset_2():
    N = 200
    sigma_squarred = 0.2
    mu = 0

    X_1 = []
    X_2 = []
    t_1 = []
    t_2 = []

    for i in range(0, int(N / 2)):
        noise_1 = np.random.normal(mu, sigma_squarred)
        noise_2 = np.random.normal(mu, sigma_squarred)
        Phi = np.random.uniform(0, np.pi)

        x_1 = np.cos(Phi) + noise_1
        y_1 = np.sin(Phi) + noise_2

        x_2 = 1 - np.cos(Phi) + noise_1
        y_2 = 1 / 2 - np.sin(Phi) + noise_2

        X_1.append(np.asarray([x_1, y_1]))
        X_2.append(np.asarray([x_2, y_2]))
        t_1.append(1)
        t_2.append(-1)

    X = np.append(X_1, X_2, 0)
    t = np.append(t_1, t_2, 0)
    return np.asarray(X), np.asarray(t).reshape(N, 1)


# creates scatter plot from data
def plotData(X, t):

    group_1 = X[np.where(t == 1)[0]]
    group_2 = X[np.where(t == -1)[0]]

    plt.scatter(group_1[:, 0], group_1[:, 1], label="t=1")
    plt.scatter(group_2[:, 0], group_2[:, 1], label="t=2")
    plt.legend()
    plt.show()



def main():
    # simple dataset
    X_1, t_1 = createDataset_1()

    # half-moon dataset
    X_2, t_2 = createDataset_2()

    # plots
    plotData(X_1, t_1)
    plotData(X_2, t_2)






if __name__ == "__main__":
    main()