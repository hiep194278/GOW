import numpy as np
import ot

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

def compute_f(function_info, x):
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

    return ValueError("Function not defined")

def compute_f_scale(function_info, x, i_scale, j_scale):
    x_scaled = x / j_scale

    match function_info[0]:
        case 'polynomial':
            return i_scale * polynomial(x_scaled, function_info[1][0], function_info[1][1])
        case 'exponential':
            return i_scale * exponential(x_scaled, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'logarithm':
            return i_scale * logarithm(x_scaled, function_info[1][0], function_info[1][1])
        case 'hyperbolic_tangent':
            return i_scale * hyperbolic_tangent(x_scaled, function_info[1][0], function_info[1][1], function_info[1][2])
        case 'polynomial_with_degree':
            return i_scale * polynomial_with_degree(x_scaled, function_info[1][0], function_info[1][1], function_info[1][2])
            
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

def gow_sinkhorn(a, b, D, function_list=[("polynomial", (1.0, 0)),], LAMBDA1=5, LAMBDA2=10, maxIter=15, stoppingCriterion='w_slope', epsilon=0.01, num_FW_iteration=100, show_details=False):
    n = D.shape[0]
    m = D.shape[1]

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    num_function = len(function_list)
    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0
    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f(function_list[k], j)

    D_ = D

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

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)

def gow_sinkhorn_autoscale(a, b, D, function_list=[("polynomial", (1.0, 0)),], LAMBDA1=5, LAMBDA2=10, maxIter=15, stoppingCriterion='w_slope', epsilon=0.01, num_FW_iteration=100, show_details=False):
    n = D.shape[0]
    m = D.shape[1]

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    num_function = len(function_list)
    i_scale = n - 1
    j_scale = m - 1
    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0
    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f_scale(function_list[k], j, i_scale, j_scale)

    w_0 = np.zeros(num_function)
    w_0[np.random.randint(num_function)] = 1
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

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)

def gow_sinkhorn_autoscale_fixed(a, b, D, LAMBDA1=10, LAMBDA2=5, maxIter=15, stoppingCriterion='w_slope', epsilon=0.01, num_FW_iteration=100, show_details=False):
    n = D.shape[0]
    m = D.shape[1]
    i_scale = n - 1
    j_scale = m - 1

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    func1 = ('polynomial_with_degree', (1.0, 1.0, 0.05))
    func2 = ('polynomial_with_degree', (1.0, 1.0, 0.28))
    func3 = ("polynomial", (1.0, 0)) 
    func4 = ('polynomial_with_degree', (1.0, 1.0, 3.2))
    func5 = ('polynomial_with_degree', (1.0, 1.0, 20))

    function_list = [func1, func2, func3, func4, func5]
    num_function = len(function_list)
    Y = np.empty((n*m, 1))
    V = np.empty((n*m, num_function))
    iterCount = 0
    F = np.empty((m, num_function))

    for j in range(m):
        for k in range(num_function):
            F[j][k] = compute_f_scale(function_list[k], j, i_scale, j_scale)

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

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)