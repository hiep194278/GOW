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
    '''
    Assuming the input function lies inside the unit square, 
    scale the function according to the lengths of 2 sequences. 
    '''

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
            
    return ValueError("Function not defined, valid function name:'polynomial', 'exponential', 'logarithm', 'hyperbolic_tangent', 'polynomial_with_degree'")

def compute_new_cost(old_D, alpha, F, LAMBDA3):
    '''
    Computing the new cost matrix with the new weight vector 
    (no scaling) 
    '''

    n = old_D.shape[0]
    m = old_D.shape[1]

    new_D = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            new_D[i][j] = old_D[i][j] + LAMBDA3 * (i - float(np.dot(np.squeeze(np.asarray(alpha)), F[j])))**2 / (n**2)

    return new_D

def compute_new_cost2(old_D, w, F, LAMBDA1):
    '''
    Computing the new cost matrix with the new weight vector 
    (the lengths of sequences are taken into account) 
    '''

    n = old_D.shape[0]
    m = old_D.shape[1]

    new_D = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            new_D[i][j] = old_D[i][j] + LAMBDA1 * (i/n - float(np.dot(np.squeeze(np.asarray(w)), F[j]))/m)**2

    return new_D

def choose_initial_w(Y, V, num_function):
    '''
    Initialize weight vector with an array that optimizes the objective GOW function. 
    The array always has an element equals to 1 and the rest of them equals to 0.
    '''

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

def gow_sinkhorn(a, b, D, function_list=[("polynomial", (1.0, 0)),], LAMBDA1=5, LAMBDA2=10, maxIter=15, epsilon=0.01, num_FW_iteration=100, show_details=False):
    '''
    Computes the GOW distance between two sequences.
    If cost matrix D is of shape n x m,
    the resulting transport matrix is of shape n x m.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain
    D : array-like, shape (dim_a, dim_b)
        loss matrix
    function_list : list, optional
        Input functions to control the warping path
    LAMBDA1 : float, optional
        Regularization term for input functions
    LAMBDA2 : float, optional
        Regularization term for Sinkhorn
    maxIter: int, optional
        Max number of iterations for the main loop (Coordinate Descent)
    epsilon: float, optional
        Stop threshold on error
    num_FW_iteration: int, optional
        Number of Frank-Wolfe iterations
    show_details: bool, optional
        Return GOW distance, transport matrix and weight vector if True

    Returns
    -------
    float : GOW distance
    float, array-like, array-like:
        GOW distance, transport matrix and weight vector (return only if show_details==True in parameters)

    '''

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
        
        # Frank-Wolfe iterations
        for FW_index in range(num_FW_iteration):
            gradient = -2 * np.matmul(np.transpose(V), Y - np.matmul(V, w_new))
            min_index = np.argmin(gradient)
            s = np.zeros((num_function, 1))
            s[min_index][0] = 1
            FW_step_size = 2 / (FW_index + 2)
            w_new = w_new + FW_step_size*(s - w_new)

        # Check stopping criterion
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

def gow_sinkhorn_autoscale(a, b, D, function_list=[("polynomial", (1.0, 0)),], LAMBDA1=5, LAMBDA2=10, maxIter=15, epsilon=0.01, num_FW_iteration=100, show_details=False):
    '''
    Computes the GOW distance between two sequences.
    If cost matrix D is of shape n x m,
    the resulting transport matrix is of shape n x m.
    The input functions will be scaled automatically
    according the sized of two sequences.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain
    D : array-like, shape (dim_a, dim_b)
        loss matrix
    function_list : list, optional
        Input functions to control the warping path
    LAMBDA1 : float, optional
        Regularization term for input functions
    LAMBDA2 : float, optional
        Regularization term for Sinkhorn
    maxIter: int, optional
        Max number of iterations for the main loop (Coordinate Descent)
    epsilon: float, optional
        Stop threshold on error
    num_FW_iteration: int, optional
        Number of Frank-Wolfe iterations
    show_details: bool, optional
        Return GOW distance, transport matrix and weight vector if True

    Returns
    -------
    float : GOW distance
    float, array-like, array-like:
        GOW distance, transport matrix and weight vector (return only if show_details==True in parameters)

    '''

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
        
        # Frank-Wolfe iterations
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
        if iterCount != 1:
            diff = (np.absolute(w_new - w_old)).max() 

            if diff < epsilon:
                break

        w_old = w_new

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)

def gow_sinkhorn_autoscale_fixed(a, b, D, LAMBDA1=10, LAMBDA2=5, maxIter=15, epsilon=0.01, num_FW_iteration=100, show_details=False):
    '''
    Computes the GOW distance between two sequences.
    If cost matrix D is of shape n x m,
    the resulting transport matrix is of shape n x m.
    Input functions are not needed because this function
    uses 5 fixed monotonic functions.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain
    D : array-like, shape (dim_a, dim_b)
        loss matrix
    LAMBDA1 : float, optional
        Regularization term for input functions
    LAMBDA2 : float, optional
        Regularization term for Sinkhorn
    maxIter: int, optional
        Max number of iterations for the main loop (Coordinate Descent)
    epsilon: float, optional
        Stop threshold on error
    num_FW_iteration: int, optional
        Number of Frank-Wolfe iterations
    show_details: bool, optional
        Return GOW distance, transport matrix and weight vector if True

    Returns
    -------
    float : GOW distance
    float, array-like, array-like:
        GOW distance, transport matrix and weight vector (return only if show_details==True in parameters)

    '''

    n = D.shape[0]
    m = D.shape[1]
    i_scale = n - 1
    j_scale = m - 1

    if len(a) == 0:
        a = np.full((n,), 1.0 / n)
    if len(b) == 0:
        b = np.full((m,), 1.0 / m)

    # 5 fixed monotonic functions
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
        
        # Frank-Wolfe iterations
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
        if iterCount != 1:
            diff = (np.absolute(w_new - w_old)).max() 

            if diff < epsilon:
                break

        w_old = w_new

    if show_details:
        return np.sum(D * T), T, w_new

    return np.sum(D * T)