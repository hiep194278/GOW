import re
import numpy as np
import ot

def compute_f(function_info, x):
    def polynomial(x, a, b):

        return a*x + b

    def exponential(x, a, b, c):

        return a * np.exp(b*x + c)

    def logarithm(x, a, b):

        return np.log(a*x + b)

    def hyperbolic_tangent(x, a, b, c):

        return a * np.tanh(b*x + c)

    def I_spline():

        return

    def polynomial_with_degree(x, a, b, c):
        return a * pow(b*x, c)

    match function_info[0]:
        case 'polynomial':
            return polynomial(x, function_info[1][0], function_info[1][1])
        case 'exponential':
            return exponential(x, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'logarithm':
            return logarithm(x, function_info[1][0], function_info[1][1])
        case 'hyperbolic_tangent':
            return hyperbolic_tangent(x, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'polynomial_with_degree':
            return polynomial_with_degree(x, function_info[1][0], function_info[1][1], function_info[1][2])
        case default:
            return ValueError("Function not defined")

def compute_new_cost(old_D, alpha, F, LAMBDA3):
    n = old_D.shape[0]
    m = old_D.shape[1]

    new_D = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            new_D[i][j] = old_D[i][j] + LAMBDA3 * (i - float(np.dot(np.squeeze(np.asarray(alpha)), F[j])))**2 / (n**2)

    return new_D

def compute_new_cost2(old_D, w, F, LAMBDA1):
    n = old_D.shape[0]
    m = old_D.shape[1]

    new_D = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            new_D[i][j] = old_D[i][j] + LAMBDA1 * (i/n - float(np.dot(np.squeeze(np.asarray(w)), F[j]))/m)**2

    return new_D

def regularization_entropy(P):
    sum = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            sum += P[i][j] * np.log(P[i][j])

    return sum

def choose_initial_w(Y, V, num_function):
    initial_w = np.zeros((num_function, 1))
    min_S = np.Inf
    min_index = 0

    for i in range(num_function):
        sub = Y - V[:,[i]]
        sum_squared = np.sum(np.square(sub))

        if sum_squared < min_S:
            min_index = i
            min_S = sum_squared

    initial_w[min_index][0] = 1

    return initial_w

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def GOW_unbalanced(a, b, D, F, num_function, LAMBDA=0.2, LAMBDA3=1, maxIter=10, stoppingCriterion='w_slope', epsilon=0.001, num_FW_iteration=100):

    n = D.shape[0]
    m = D.shape[1]

    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    D_ = D
    iterCount = 0

    while iterCount < maxIter:
        
        iterCount = iterCount + 1
        # print("Loop:", iterCount)

        # Optimization for T
        T = ot.unbalanced.mm_unbalanced(a, b, D_, LAMBDA, 'kl')

        # Optimization for w
        index_Y = 0
        for i in range(n):
            for j in range(m):
                temp = np.sqrt(T[i][j])
                Y[index_Y][0] =  temp * i
                           
                for k in range(num_function):
                    V[index_Y][k] = temp * F[j][k]
                
                index_Y = index_Y + 1

        w_new = choose_initial_w(Y, V, num_function)
        
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

        # Check stopping criterion
        match stoppingCriterion:
            case 'alpha_slope': 
                if iterCount != 1:
                    diff = (np.absolute(w_new - w_old)).max() 
                    # diff = np.sqrt(np.sum((w_new - w_old) ** 2))
                    
                    # print('Difference: ', diff)

                    if diff < epsilon:
                        break


        # New cost matrix from new w
        D_ = compute_new_cost2(D, w_new, F, LAMBDA3)

        w_old = w_new

    return D_, T, w_new

def GOW_sinkhorn(a, b, D, F, num_function, LAMBDA1=2, LAMBDA2=5, maxIter=10, stoppingCriterion='w_slope', epsilon=0.001, num_FW_iteration=100):

    n = D.shape[0]
    m = D.shape[1]

    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    D_ = D
    iterCount = 0

    while iterCount < maxIter:
        
        iterCount = iterCount + 1

        # Optimization for T
        T = ot.sinkhorn(a, b, D_, 1.0/LAMBDA2)

        # Optimization for w
        index_Y = 0
        for i in range(n):
            for j in range(m):
                temp = np.sqrt(T[i][j])
                Y[index_Y][0] =  temp * i
                           
                for k in range(num_function):
                    V[index_Y][k] = temp * F[j][k]
                
                index_Y = index_Y + 1

        w_new = choose_initial_w(Y, V, num_function)
        
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

        # Check stopping criterion
        match stoppingCriterion:
            case 'alpha_slope': 
                if iterCount != 1:
                    diff = (np.absolute(w_new - w_old)).max() 
                    # diff = np.sqrt(np.sum((w_new - w_old) ** 2))

                    if diff < epsilon:
                        break


        # New cost matrix from new w
        D_ = compute_new_cost2(D, w_new, F, LAMBDA1)

        w_old = w_new

    return D_, T, w_new

def GOW_sinkhorn_autoscale(a, b, D, LAMBDA1=2, LAMBDA2=5, maxIter=10, stoppingCriterion='w_slope', epsilon=0.001, num_FW_iteration=100, return_convergence_array=False):

    if return_convergence_array:
        convergence_array = []

    n = D.shape[0]
    m = D.shape[1]

    i_scale = n - 1
    j_scale = m - 1

    func1 = ('polynomial_with_degree', (1.0*i_scale, 1.0/j_scale, 0.05))
    func2 = ('polynomial_with_degree', (1.0*i_scale, 1.0/j_scale, 0.28))
    func3 = ("polynomial", (1.0*i_scale/j_scale, 0)) 
    func4 = ('polynomial_with_degree', (1.0*i_scale, 1.0/j_scale, 3.2))
    func5 = ('polynomial_with_degree', (1.0*i_scale, 1.0/j_scale, 20))

    function_list = [func1, func2, func3, func4, func5]

    num_function = len(function_list)

    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0

    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f(function_list[k], j)

    w_0 = np.array([0, 0, 1, 0, 0])
    D_ = compute_new_cost(D, w_0, F, LAMBDA1)

    while iterCount < maxIter:
        
        iterCount = iterCount + 1

        # Optimization for T
        T = ot.sinkhorn(a, b, D_, 1.0/LAMBDA2)

        # Optimization for w
        index_Y = 0
        for i in range(n):
            for j in range(m):
                temp = np.sqrt(T[i][j])
                Y[index_Y][0] =  temp * i
                           
                for k in range(num_function):
                    V[index_Y][k] = temp * F[j][k]
                
                index_Y = index_Y + 1

        w_new = choose_initial_w(Y, V, num_function)
        
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

        if return_convergence_array:
            convergence_array.append(np.sum(D_ * T) + (1.0/LAMBDA2) * regularization_entropy(T))
        
        # New cost matrix from new w
        D_ = compute_new_cost(D, w_new, F, LAMBDA1)

        # Check stopping criterion
        match stoppingCriterion:
            case 'w_slope': 
                if iterCount != 1:
                    diff = (np.absolute(w_new - w_old)).max() 

                    if diff < epsilon:
                        break

        w_old = w_new

    print("Iterations:", iterCount)

    if return_convergence_array:
        return D_, T, w_new, convergence_array

    return D_, T, w_new