import numpy as np
import matplotlib.pyplot as plt

num_train_points = 200
epsilon = 0.03
Iterations = 500
gram_matrix_simple = np.zeros((num_train_points,num_train_points))
gram_matrix_halfmoon = np.zeros((num_train_points,num_train_points))

##########################
## TASK 1
#
# draw data for simple dataset
def create_simple_dataset():
    N = num_train_points
    mu_1 = [6.5, 2]
    mu_2 = [1.5, 2]
    Sigma_1 = [[0.8, 0],
               [0, 0.7]]
    Sigma_2 = [[1, 0],
               [0, 0.2]]
    X = []
    t = []
    
    for i in range(0, int(N / 2)):
        value = np.random.multivariate_normal(mu_1, Sigma_1)
        X.append(value)
        t.append(1)

    for i in range(0, int(N / 2)):
        X.append(np.random.multivariate_normal(mu_2, Sigma_2))
        t.append(-1)
    return np.asarray(X), np.asarray(t).reshape(N, 1)

# draw data for halfmoon dataset
def create_halfmoon_dataset():
    N = num_train_points
    sigma = 0.1
    mu = 0

    X_1 = []
    X_2 = []
    t_1 = []
    t_2 = []

    for i in range(0, int(N / 2)):
        noise_1 = np.random.normal(mu, sigma)
        noise_2 = np.random.normal(mu, sigma)
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

        indices, offset, margin = get_supprot_vectors_and_margins(X, t, w_tilde, k, d)
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

    
##########################
## TASK 2
#
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
def proximal_subgradient_method(X, t):
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

# calculates classification calculate_accuracy
def calculate_accuracy(X, t, w_tilde):
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
def get_supprot_vectors_and_margins(X, t, w_tilde, k, d):
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
    
##########################
## TASK 3
#
# calculates the gaussian kernel, formula taken from assignment sheet
def calculate_gaussian_kernel(x_n, x_m):
    #TODO: 4c) play with it
    sigma_squarred = 2
    #sigma_squarred = 1.4

    distance_norm = np.linalg.norm(x_n - x_m)    
    return np.exp(-(distance_norm**2) / (2 * sigma_squarred))

# calculates the gram matrix for the simple dataset    
def create_gram_matrix_simple(x, t):
    for i in range(num_train_points):
        for j in range(num_train_points):               
            gram_matrix_simple[j][i] = t[i] * t[j] * calculate_gaussian_kernel(x[j], x[i])
            #print("t"+str(i+1) + " t" + str(j+1))
    print("Gram Matrix created for simple dataset.")

# calculates the gram matrix for the halfmoon dataset    
def create_gram_matrix_halfmoon(x, t):
    for i in range(num_train_points):
        for j in range(num_train_points):
            gram_matrix_halfmoon[j][i] = t[i] * t[j] * calculate_gaussian_kernel(x[j], x[i])
            #print("t"+str(i+1) + " t" + str(j+1))
    print("Gram Matrix created for halfmoon dataset.")     

    
# fista calculates optimal dual variable a
# algorithm taken from assignment sheet   
def fista(gram_matrix, alpha):
    t = 1
    a = np.zeros((num_train_points,1))
    a_prev = np.zeros((num_train_points,1))
    # variable which is selected manually

    a_list = []

    # variable which is selected manually
    max_i = 1500
    #max_i = 5000
    
    for i in range(1,max_i+1):
        t_next = (1 + np.sqrt(1 + 4*t*t)) / 2
        a_tilde = a + ((t - 1)/t_next) * (a - a_prev)
        a_next = proj(a_tilde + alpha * compute_gradient(a_tilde, gram_matrix))
        
        a_list.append(a)
        
        #assign new values for next iteration
        t = t_next
        a_prev = a
        a = a_next

    return a_list   

# computes gradient of d(a). our derivation can be found in our report
def compute_gradient(a_tilde, gm):
    i = np.ones((num_train_points,1))
    return  i - np.matmul(gm, a_tilde)

# replaces all values below 0 in a vector with 0
# algorithm taken from assignment sheet
def proj(alpha):
    return alpha.clip(0)
   
# computes the dual energy over the course of the optimization   
def compute_dual_energy(a_list, gram_matrix):
    energy_list = []
    i = np.asmatrix(np.ones((num_train_points,1)))
    for variable in a_list:
        a = np.asmatrix(variable)
        energy = i.transpose() * a - 0.5 * (a.transpose() * gram_matrix * a)
        
        energy_list.append(energy)   
    return np.squeeze(np.asarray(energy_list))

# plots this dual energy    
def plot_dual_energy(energy):
    num_iter = energy.shape[0]
    
    x = np.arange(1,num_iter+1)
    plt.plot(x, energy)
    plt.xlabel("iteration over time")
    plt.ylabel("energy")
    plt.title(f"Dual energy D(*a*) over {num_iter} iterations of the optimization")
    plt.show()


# plots the decision bounday together with the margins and the support vectors    
def plot_decision_boundary(data_points, labels, opt_a, plot_title):
    # inspired by: 
    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
    max_value = int(np.max(np.ceil(data_points[:,0])))
    min_value = int(np.min(np.floor(data_points[:,0])))
    u = np.linspace(min_value, max_value, 100)
    x, y = np.meshgrid(u,u)
    svx, svy = get_support_vectors(np.asmatrix(data_points), labels, opt_a)
    z = get_predictions(x, y, np.asmatrix(data_points), labels, opt_a)    
    
    fig2, ax2 = plt.subplots(constrained_layout=True)
    cs = ax2.contour(x, y, z, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    #cs.cmap.set_over('red')
    #cs.cmap.set_under('blue')
    
    labels = ["db -1", "decision boundary", "db +1"]
    for i in range(3):
        cs.collections[i].set_label(labels[i])

    plt.title(plot_title)
    plt.xlabel("x coordinates of points")
    plt.ylabel("y coordinates of points")
    plt.scatter(data_points[:, 0], data_points[:, 1], marker="x", label="data points")
    plt.scatter(svx, svy, s=60, edgecolors="r", facecolors="none", label="support vectors")
    plt.legend()
    
    plt.show()

# calculates the predictions for a certain points. formula taken from assignment sheet    
def get_predictions(x, y, data_points, labels, opt_a):
    z = np.zeros((100,100))
    ones = np.ones((1,num_train_points))
    for z_y in range(0,100):
        for z_x in range(0,100):
            point_x = x[z_y][z_x]
            point_y = y[z_y][z_x]
            point = np.array([point_x, point_y])
            
            sum = 0.0
            for i in range(num_train_points):
                iter = (opt_a[i] * labels[i] * calculate_gaussian_kernel(point, data_points[i]))
                sum = sum + iter
            
            z[z_y][z_x] = sum
        
        if(z_y % 5 == 0):
            print(f"Iteration {z_y}/100.")
    return z

# returns the support vectors    
def get_support_vectors(data_points, labels, opt_a):
    support_vectors_x = []
    support_vectors_y = []
    
    for dp in range(num_train_points):
        sum = 0.0
        for i in range(num_train_points):
                iter = (opt_a[i] * labels[i] * calculate_gaussian_kernel(data_points[dp], data_points[i]))
                #iter = calculate_gaussian_kernel(point, data_points[i])
                sum = sum + iter
        
        if(sum[0,0] < 1+epsilon and sum[0,0] > 1-epsilon):
            support_vectors_x.append(data_points[dp][0,0])
            support_vectors_y.append(data_points[dp][0,1])
        if(sum[0,0] > -1-epsilon and sum[0,0] < -1+epsilon):
            support_vectors_x.append(data_points[dp][0,0])
            support_vectors_y.append(data_points[dp][0,1])    

    return support_vectors_x, support_vectors_y

# creates title for plot
def get_simple_title():
    curr_dataset = "Simple dataset"
    sigma = 2
    alpha = 0.0001
    
    return f"{curr_dataset} with decision boundary and margins; sigma squarred {sigma}, alpha: {alpha}"

# creates title for plot
def get_halfmoon_title():
    curr_dataset = "Half-Moon dataset"
    sigma = 1.4
    alpha = 0.1
    
    return f"{curr_dataset} with decision boundary and margins; sigma squarred {sigma}, alpha: {alpha}"

    
##########################
## MAIN
#
def main():
    ###########
    ## TASK 1
    #
    
    x_simple, t_simple = create_simple_dataset()
    x_halfmoon, t_halfmoon = create_halfmoon_dataset()

    # plots
    plotData(x_simple, t_simple)
    plotData(x_halfmoon, t_halfmoon)
    
    ###########
    ## TASK 2
    #
    
    result_simple = proximal_subgradient_method(x_simple, t_simple)
    calculate_accuracy_1 = calculate_accuracy(x_simple, t_simple, result_simple)
    plotData(x_simple, t_simple, result_simple)
    
    '''
    result_halfmoon = proximal_subgradient_method(x_halfmoon, t_halfmoon)
    calculate_accuracy_2 = calculate_accuracy(x_halfmoon, t_halfmoon, result_halfmoon)
    plotData(x_halfmoon, t_halfmoon, result_halfmoon)
    '''
    
    
    ###########
    ## TASK 3
    #
    
    # simple dataset experiments
    create_gram_matrix_simple(x_simple, t_simple)
    computed_dual_variables = fista(np.asmatrix(gram_matrix_simple),0.0001)
    dual_energy_over_optimization = compute_dual_energy(computed_dual_variables, np.asmatrix(gram_matrix_simple))
    plot_dual_energy(dual_energy_over_optimization)
    
    #get optimal a
    opt_a_idx = np.argmax(dual_energy_over_optimization)
    opt_a = computed_dual_variables[opt_a_idx]
    # plots with decision boundary and margin
    plot_decision_boundary(x_simple, t_simple, opt_a, get_simple_title())
    
    '''
    # don't forget to change sigma^2 in function calculate gaussian kernel if you want to check out the function!
    # if seperation task is too hard (= overlapping decision regions) won't work tho
    #half-moon dataset
    create_gram_matrix_halfmoon(x_halfmoon, t_halfmoon)
    computed_dual_variables = fista(np.asmatrix(gram_matrix_halfmoon), 0.01)
    dual_energy_over_optimization = compute_dual_energy(computed_dual_variables, np.asmatrix(gram_matrix_halfmoon))
    plot_dual_energy(dual_energy_over_optimization)
    
    #get optimal a
    opt_a_idx = np.argmax(dual_energy_over_optimization)
    opt_a = computed_dual_variables[opt_a_idx]
    #plots with decision boundary and margin
    plot_decision_boundary(x_halfmoon, t_halfmoon, opt_a, get_halfmoon_title())
    '''
    

if __name__ == "__main__":
    main()