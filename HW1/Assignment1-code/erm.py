import numpy as np
import matplotlib.pyplot as plt


PI = np.pi
SIGMA_SQUARED = 0.1
N = 100


def createData():
    X = [np.random.random() * 2 * PI for i in range(0, N)]
    t = []
    for sample in X:
        t.append(np.sin(sample) + np.random.normal(0, np.sqrt(SIGMA_SQUARED)))
    return X, t


def createTheta(X, size):
    theta = np.zeros((N, size + 1))
    for i in range(0, N):
        xi = X[i]
        for j in range(0, size + 1):
            theta[i][j] = xi ** j
    return theta


def y(x, w):
    result = 0.
    exponent = 0
    for weight in w:
        result += weight * (x ** exponent)
        exponent += 1
    return result


def plot(weights, X, t, empirical_risk, true_risk):
    deg = 1
    # plot different polynomials with varying degree
    for w in weights:
        plt.ylim(-1.5, 1.5)
        x_axis = np.arange(0, 2 * PI, 0.01)
        y_axis = y(x_axis, w)
        y_sin = np.sin(x_axis)
        plt.plot(x_axis, y_axis, label="ŷ(x) (degree = " + str(deg) + ")")
        plt.plot(x_axis, y_sin, label="y*(x) (=sin(x))")
        deg += 1

        plt.scatter(X, t, c='C2', label="Data Samples")
        plt.xlabel("X")
        plt.ylabel("y(x)")
        plt.title("Model of degree " + str(deg) + "(N =  " + str(N) + ")")
        plt.legend()
        plt.show()
    
    # plot empirical risk
    plt.bar(np.arange(1, 9, 1), empirical_risk)
    plt.title("Empirical Risk over Increasing Model Complexity (N = " + str(N) + ")")
    plt.xlabel("Model Complexity")
    plt.ylabel("Empirical Risk")
    plt.show()

    # plot true risk
    plt.bar(np.arange(1, 9, 1), true_risk)
    plt.title("True Risk over Increasing Model Complexity (N = " + str(N) + ")")
    plt.xlabel("Model Complexity")
    plt.ylabel("True Risk")
    plt.yscale("log")
    plt.show()


def empiricalRisk(X, t, weights):
    risk = []
    for w in weights:
        risk.append(0)
        for ti, xi in zip(t, X):
            risk[-1] += (np.abs(y(xi, w) - ti) ** 2)

        risk[-1] /= N

    return risk


def trueRisk(X, t, weights):
    risk = []
    for w in weights:
        risk.append(0)
        for xi in np.arange(0, 2 * PI, 0.01):
            risk[-1] += (y(xi, w) - np.sin(xi)) ** 2
        for ti, xi in zip(t, X):
            risk[-1] += (np.abs(np.sin(xi) - ti) ** 2)

        risk[-1] /= N
    return risk


def main():
    X, t = createData()
    weights = []

    for i in range(1, 9):
        theta = createTheta(X, i)
        weights.append(np.matmul(np.linalg.pinv(theta), t))

    empirical_risk = empiricalRisk(X, t, weights)
    true_risk = trueRisk(X, t, weights)

    plot(weights, X, t, empirical_risk, true_risk)






if __name__ == "__main__":
    main()
