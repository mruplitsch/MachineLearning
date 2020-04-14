import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def gaussian2D(pos, mu, sigma):
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** 2 * sigma_det)
    fac = np.dot(pos - mu, sigma_inv)
    fac = np.einsum('ijk,ijk->ij', fac, pos - mu)

    return np.exp(-fac / 2) / N

def plotGaussian2D(N, mu, sigma, alpha):
    # Our 2-dimensional distribution will be over variables X and Y
    X = np.linspace(-3.5, 3.5, N)
    Y = np.linspace(-3.5, 3.5, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = gaussian2D(pos, mu, sigma)

    # Create plot
    fig = plt.figure()
    fig.suptitle("2D Gaussian (alpha = " + str(alpha) + ")")
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=1, antialiased=True, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(0, 0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))
    ax.view_init(40, -20)

    plt.show()


def plotGaussianMarginals(N, mu, sigma, alpha, variable_counter):
    print("hi")
    X = np.linspace(-3.5, 3.5, N)
    Y = np.exp(-0.5 * ((X - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    plt.plot(X, Y, linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gaussian Marginal X" + str(variable_counter) + "(alpha = " + str(alpha) + ")")
    plt.show()


def main():
    # input parameters
    N = 100
    alpha = 0.5
    mu = np.array([0., 0.])
    sigma = np.array([[1., alpha], [alpha,  1]])

    X = np.linspace(-3.5, 3.5, N)

    plotGaussian2D(N, mu, sigma, alpha)
    plotGaussianMarginals(N, mu[0], sigma[0][0], alpha, 1)
    plotGaussianMarginals(N, mu[1], sigma[1][1], alpha, 2)


if __name__ == "__main__":
    main()