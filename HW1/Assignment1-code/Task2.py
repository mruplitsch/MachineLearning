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

def plotGaussian2D(N, mu, sigma, alpha, sigma_squared):
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
    fig.suptitle("Bivariate Gaussian (alpha = " + str(alpha) + ", sigma^2 = " + str(sigma_squared) + ")")
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=1, antialiased=True, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(0, 0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))
    ax.view_init(40, -20)

    plt.savefig("Bivariate Gaussian (alpha = " + str(alpha) + ", sigma^2 = " + str(sigma_squared) + ").png")
    plt.show()


def plotGaussianMarginals(N, mu, sigma, alpha, sigma_squared, variable_counter):
    X = np.linspace(-3.5, 3.5, N)
    Y = np.exp(-0.5 * ((X - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    plt.plot(X, Y, linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gaussian Marginal X" + str(variable_counter) + "(alpha = " + str(alpha) + ", sigma^2 = " + str(sigma_squared) + ")")
    plt.savefig("X" + str(variable_counter) + "(alpha = " + str(alpha) + ", sigma^2 = " + str(sigma_squared) + ").png")
    plt.show()


def main():
    # input parameters
    N = 100
    mu = np.array([0., 0.])
    alphas = [0.2, -0.5, 0.8]
    sigmas = [0.5, 1, 1.5]

    for alpha in alphas:
        for sigma_squared in sigmas:
            sigma = np.array([[sigma_squared, alpha * sigma_squared], [alpha * sigma_squared,  sigma_squared]])

            plotGaussian2D(N, mu, sigma, alpha, sigma_squared)
            plotGaussianMarginals(N, mu[0], sigma[0][0], alpha, sigma_squared, 1)
            plotGaussianMarginals(N, mu[1], sigma[1][1], alpha, sigma_squared, 2)


if __name__ == "__main__":
    main()