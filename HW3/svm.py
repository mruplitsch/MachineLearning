import numpy as np
import matplotlib.pyplot as plt

Iterations = 500


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

    return np.asarray(X), np.reshape(t, (N, 1))


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
    return np.asarray(X), np.reshape(t, (N, 1))


# phi function from Assignment Sheet
def phi(x):
    return np.append(1, x)


# g_i function from Assignment Sheet
def g_i(w_tilde, xn, tn):
    if tn * np.inner(w_tilde, phi(xn)) >= 1:
        return 0
    else:
        return -tn * phi(xn)


# algorithm 1 from Assignment Sheet
def proximalSubGradientMethod(X, t):
    N = np.shape(t)[0]
    gamma_1 = 10 ** (-6)
    gamma = np.array(([gamma_1, 1.0, 1.0]))
    w_tilde = [np.array(([1, 1, 1]))]
    alpha = 0.01
    lambda_ = 0.01

    for j in range(0, Iterations):
        if(j % 50 == 0):
            print("Iteration " + str(j) + "/" + str(Iterations))
        gs = []
        for i, xn, tn in zip(range(0, N), X, t[:, 0]):
            gs.append(g_i(w_tilde[-1], xn, tn))
            g = np.sum(gs) / (i + 1)

            next_w_tilde = (w_tilde[-1] - alpha * g) / (1 + alpha * lambda_ * gamma)
            w_tilde.append(next_w_tilde)
    return w_tilde[-1]


# calculates classification accuracy
def accuracy(X, t, w_tilde):
    N = np.shape(X)[0]
    correct = 0
    for xn, tn in zip(X, t):
        y = np.inner(w_tilde, phi(xn))
        if(y >= 0 and tn == +1):
            correct += 1
        elif (y < 0 and tn == -1):
            correct += 1
    return correct / N


# finds data points closest to separating hyperplane
# and calculates y offset from hyperplane
def supportVector(X, t, w_tilde, k, d):
    N = np.shape(X)[0]
    indices = []
    offset = 0
    min = np.abs(np.inner(w_tilde, phi(X[0])))
    margin = np.abs(w_tilde[0] + d * w_tilde[2]) / np.sqrt((w_tilde[1] ** 2) + (w_tilde[2] ** 2))
    for i, xn, tn in zip(range(0, N), X, t):
        y = np.abs(np.inner(w_tilde, phi(xn)))
        if y < min:
            indices = [i]
            offset = k * xn[0] + d - xn[1]

            min = y
            d_2 = d - offset
            margin = np.abs((d - offset) * w_tilde[2] + w_tilde[0]) / np.sqrt((w_tilde[1] ** 2) + (w_tilde[2] ** 2))
        elif y == min:
            indices.append(i)

    return indices, offset, margin


# creates scatter plot from data
def plotData(X, t, w_tilde=[]):
    plt.title("Data Set With N = 200")

    group_1 = X[np.where(t == 1)[0]]
    group_2 = X[np.where(t == -1)[0]]

    plt.scatter(group_1[:, 0], group_1[:, 1], label="t = +1")
    plt.scatter(group_2[:, 0], group_2[:, 1], label="t = -1")

    if len(w_tilde):
        d = w_tilde[0] / -w_tilde[2]
        k = w_tilde[1] / -w_tilde[2]
        x = np.arange(-3, 10, 0.1)
        y = k * x + d
        plt.plot(x, y, 'C3', label="Separating Hyperplane")

        indices, offset, margin = supportVector(X, t, w_tilde, k, d)
        plt.plot(x, k * x + d - offset, 'C5', label="Support Vectors")
        plt.plot(x, k * x + d + offset, 'C5')
        plt.scatter(X[indices, 0], X[indices, 1])
        plt.title("Proximal Sub-gradient Method with " + str(Iterations) + " Iterations, Margin = {:.2f}".format(margin))

    xmin = np.min(X[:, 0]) - 1
    ymin = np.min(X[:, 1]) - 1
    xmax = np.max(X[:, 0]) + 1
    ymax = np.max(X[:, 1]) + 1
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.legend()
    plt.show()



def main():
    # Task 1
    # simple dataset
    X_1, t_1 = createDataset_1()

    # half-moon dataset
    X_2, t_2 = createDataset_2()

    # Task 2
    result_1 = proximalSubGradientMethod(X_1, t_1)
    result_2 = proximalSubGradientMethod(X_2, t_2)

    accuracy_1 = accuracy(X_1, t_1, result_1)
    accuracy_2 = accuracy(X_2, t_2, result_2)

    print("Accuracies:")
    print(accuracy_1)
    print(accuracy_2)

    # plots
    plotData(X_1, t_1)
    plotData(X_2, t_2)

    # plots with decision boundary
    plotData(X_1, t_1, result_1)
    plotData(X_2, t_2, result_2)






if __name__ == "__main__":
    main()