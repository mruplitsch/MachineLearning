import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

Sigma_1 = np.array([[0.8, 0.0],
                    [0.0, 0.7]])
Sigma_2 = np.array([[1.0, 0.0],
                    [0.0, 0.2]])
mu_1 = np.array([6.5, 2])
mu_2 = np.array([0.5, 2])
iterations = 100


# calculates lipschitz constant with formula from assignment sheet
def lipschitz_constant(x, s_squared):
    sum = np.zeros((np.shape(x)[1], np.shape(x)[1]))
    sum.shape = (np.shape(x)[1], np.shape(x)[1])
    for xn in x:
        sum += np.matmul(xn[None].T, xn[None])

    sigma_max = np.max(sum)
    L = (1 / 4) * sigma_max + 1 / s_squared
    return L


# calculate p(x) = N(x | mu, Sigma)
def bivariate_gaussian(x, mu, Sigma, D):
    Sigma_inv = np.linalg.pinv(Sigma)
    result = np.exp(-0.5 * np.matmul(np.matmul((x - mu).T, Sigma_inv), x - mu))
    result *= 1 / np.sqrt(((2 * np.pi) ** D) * np.linalg.norm(Sigma))
    return result


# logistic sigmoid
def logistic_sigmoid(a):
    return 1 / (1 + np.exp(-a))


# derivative of above function
def logistic_sigmoid_derivative(a):
    return np.exp(-a) / ((1 + np.exp(-a)) ** 2)


# energy = log-likelihood
def energy(w_tilde, x, t, s_squared):
    D = np.shape(x)[1]
    w = w_tilde[1:]
    mu = np.zeros(D - 1)
    Sigma = s_squared * np.diag(np.ones(D - 1))
    sum = 0
    for xn, tn in zip(x, t):
        sum += np.log(logistic_sigmoid(tn * np.matmul(w_tilde.T, xn)))
    sum += np.log(bivariate_gaussian(w, mu, Sigma, D))
    return -sum


# implements the gradient of the energy calculated by hand
def energy_gradient(w_tilde, x, t, s_squared):
    vec = w_tilde[1:]
    vec = np.append(0, vec)
    gradient = (1 / s_squared) * vec
    for xn, tn in zip(x, t):
        foo = 1 / (1 + np.exp(tn * np.matmul(w_tilde.T, xn)))
        foo *= np.exp(tn * np.matmul(w_tilde.T, xn))
        foo = foo * xn * tn
        gradient += foo

    return gradient


# draws random samples from two 2D normal distributions
def generate_data(N):
    x_1 = [np.random.multivariate_normal(mu_1, Sigma_1) for i in range(0, int(N / 2))]
    x_2 = [np.random.multivariate_normal(mu_2, Sigma_2) for i in range(0, int(N / 2))]
    x = np.vstack((np.array(x_1), np.array(x_2)))
    t = np.array([1] * int(N / 2))
    t = np.append(t, np.array([-1] * int(N / 2)))
    t = t.reshape((N, 1))

    return x, t


# plots contour lines of sigmoid function
def surface_plot(w_tilde, s_squared):
    x_linspace = np.linspace(-3, 10, 100)
    y_linspace = np.linspace(-2, 5)
    xx, yy = np.meshgrid(x_linspace, y_linspace)
    zz = logistic_sigmoid(np.matmul(w_tilde.T, [1, xx, yy]))
    fig, ax = plt.subplots()
    CS = ax.contour(xx, yy, zz, levels=np.arange(0.1, 1, 0.1))
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title("Contour Plot (s^2 = " + str(s_squared) + ")")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# scatter plot with optional parameter to draw decision boundary
def plot(x, t, s_squared, w=None):
    x_1 = x[np.where(t == 1)[0]]
    x_2 = x[np.where(t == -1)[0]]
    plt.scatter(x_1[:, 0], x_1[:, 1], label="Class 1")
    plt.scatter(x_2[:, 0], x_2[:, 1], label="Class 2")
    plt.legend()

    if not(w is plot.__defaults__[0]):
        # transform w so that we can plot it more easily
        # y = kx + d
        d = -w[0] / w[2]
        k = -d / (-w[0] / w[1])
        grid = np.linspace(-3, 10, 100)

        y = [d + k * x for x in grid]
        plt.plot(grid, y, c="g")
    plt.title("Scatter plot (s^2 = " + str(s_squared) + ")")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim((-3, 10))
    plt.ylim((-2, 5))
    plt.show()


def plot_accuracies(s_squared, accuracies, training):
    if training:
        set = "Training"
    else:
        set = "Validation"
    plt.bar(np.arange(len(s_squared)), accuracies)
    plt.xticks(np.arange(len(s_squared)), s_squared)
    plt.title(set + " Accuracies with different s^2")
    plt.ylabel("Accuracy")
    plt.xlabel("s^2")
    plt.show()


# numerically computes gradient of energy and compares to exact gradient calculated by hand
def check_with_scipy(x, t, s_squared, w_tilde=0):
    if w_tilde == 0:
        w_tilde = np.random.rand(3)
    print("Comparing Numerical Solution to Exact Solution")
    numerical_solution = approx_fprime(w_tilde, energy, 1e-6, x, t, s_squared)
    print("Numerical  Solution      : " + str(numerical_solution))

    exact_solution = energy_gradient(w_tilde, x, t, s_squared)
    print("Exact Solution           : " + str(exact_solution))


# training algorithm provided in assignment sheet
# takes design matrix and target values and outputs weights
def nesterov_gradient_method(x, t, s_squared):
    D = np.shape(x)[1]
    w_tilde = np.random.rand(D)
    L = lipschitz_constant(x, s_squared)
    w = np.zeros((iterations + 1, D))
    w[0] = w_tilde
    w[1] = w_tilde
    for k in range(1, iterations):
        if k % 10 == 0:
            print("Progress: " + str(k) + " / " + str(iterations))

        beta = (k - 1) / (k + 1)
        w_dash = w[k] + beta * (w[k] - w[k - 1])
        #w[k + 1] = w_dash - approx_fprime(w_dash, energy, 1e-6, x, t, s_squared) / L
        w[k + 1] = w_dash - energy_gradient(w_dash, x, t, s_squared) / L
    return -w[-1]


# takes weights and design matrix and compares predicted results to actual results
def predict(x, t, w):
    N = np.shape(x)[0]
    accuracy = 0
    # predict target classes for testing data and calculate accuracy
    for xn, tn in zip(x, t):
        prediction = logistic_sigmoid(np.matmul(w.T, xn))
        if prediction >= 0.5 and tn == 1:
            accuracy += 1
        elif prediction < 0.5 and tn == -1:
            accuracy += 1
    accuracy /= N
    return accuracy


def task_5():
    print("Starting Task 5")
    # initialize all variables
    s_squared = 0.1
    N = 1000
    x, t = generate_data(N)
    x_tilde = np.hstack((np.ones((N, 1)), x))
    check_with_scipy(x_tilde, t, s_squared)

    # run training algorithm to generate weights
    result = nesterov_gradient_method(x_tilde, t, s_squared)
    # generate plots
    plot(x, t, s_squared)
    plot(x, t, s_squared, result)
    surface_plot(result, s_squared)


def task_6():
    print("Starting Task 6")
    # load training data and modify to our needs
    x = np.load("spam_train.npy")
    t = x[:, -1]
    x = x[:, :-1]
    x_tilde = np.hstack((np.ones((np.shape(x)[0], 1)), x))

    # load testing data and modify to our needs
    x_test = np.load("spam_val.npy")
    t_test = x_test[:, -1]
    x_test = x_test[:, :-1]
    x_test_tilde = np.hstack((np.ones((np.shape(x_test)[0], 1)), x_test))

    # run training several times with different hyper-parameter s_squared
    accuracies_training = []
    accuracies_validation = []
    params = [0.1, 1, 10, 100, 1000]
    for s in params:
        print("training with s_squared = " + str(s))
        # run training algorithm to generate weights
        result = nesterov_gradient_method(x_tilde, t, s)

        # calculate accuracy
        accuracies_training.append(predict(x_tilde, t, result))
        accuracies_validation.append(predict(x_test_tilde, t_test, result))

    # plot accuracies
    print(accuracies_training)
    print(accuracies_validation)
    plot_accuracies(params, accuracies_training, True)
    plot_accuracies(params, accuracies_validation, False)



def main():
    task_5()
    task_6()






if __name__ == "__main__":
    main()