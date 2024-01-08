import java.security.InvalidAlgorithmParameterException;
import org.jblas.DoubleMatrix;
import java.util.Arrays;

class Monotonic_Function {
    String name;
    double[] parameters;

    public Monotonic_Function(String name_, double[] parameters_) {
        name = name_;
        parameters = parameters_;
    }

    private static double polynomial(double x, double a, double b) {
        return a*x + b;
    }

    private static double exponential(double x, double a, double b, double c) {
        return a * Math.exp(b*x + c);
    }

    private static double logarithm(double x, double a, double b) {
        return Math.log(a*x + b);
    }

    private static double hyperbolic_tangent(double x, double a, double b, double c) {
        return a * Math.tanh(b*x + c);
    }

    private static double polynomial_with_degree(double x, double a, double b, double c) {
        return a * Math.pow(b*x, c);
    }

    public static double compute_f(Monotonic_Function function_info, double x) throws InvalidAlgorithmParameterException {
        switch(function_info.name) {
            case "polynomial":
                return polynomial(x, function_info.parameters[0], function_info.parameters[1]);
            case "exponential":
                return exponential(x, function_info.parameters[0], function_info.parameters[1], function_info.parameters[2]);
            case "logarithm":
                return logarithm(x, function_info.parameters[0], function_info.parameters[1]);
            case "hyperbolic_tangent":
                return hyperbolic_tangent(x, function_info.parameters[0], function_info.parameters[1], function_info.parameters[2]);
            case "polynomial_with_degree":
                return polynomial_with_degree(x, function_info.parameters[0], function_info.parameters[1], function_info.parameters[2]);
            default:
                throw new InvalidAlgorithmParameterException("Invalid function name. Valid names: \"polynomial\", \"exponential\", \"logarithm\", \"hyperbolic_tangent\"");                                                                                   
        }
    }
}

class MM_Unbalanced {
    private DoubleMatrix a, b;
    private String div;
    private int numIterMax, n, m;
    private double reg_m, stopThr;

    public MM_Unbalanced(DoubleMatrix a_, DoubleMatrix b_, double reg_m_, String div_) {
        a = a_;
        b = b_;
        n = a.length;
        m = b.length;
        reg_m = reg_m_;
        numIterMax = 1000;
        stopThr = 1e-15;
        div = div_;
    }

    public DoubleMatrix unbalanced_ot_dist(DoubleMatrix M) {
        DoubleMatrix G = a.mmul(b.transpose());
        DoubleMatrix K = new DoubleMatrix(n, m);

        if (div.equals("kl")) {
            K = M.div(-reg_m).div(2);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    K.put(i, j, Math.exp(K.get(i, j)));
                }
            }
        } else if (div.equals("l2")) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    double maxElement = Math.max(a.get(i) + b.get(j) - M.get(i, j) / reg_m / 2, 0);
                    K.put(i, j, maxElement);
                }
            }
        }

        for (int i = 0; i < numIterMax; i++) {
            DoubleMatrix G_prev = G;

            if (div.equals("kl")) {
                DoubleMatrix u = a.div(G.rowSums().add(1e-16));

                for (int j = 0; j < n; j++)
                    u.put(j, Math.sqrt(u.get(j)));

                DoubleMatrix v = b.div(G.columnSums().add(1e-16));

                for (int j = 0; j < m; j++)
                    v.put(j, Math.sqrt(v.get(j)));

                G = G.mul(K.mul(u.mmul(v.transpose())));

            } else if (div.equals("l2")) {

                DoubleMatrix u = G.rowSums(), v = G.columnSums();

                DoubleMatrix Gd = new DoubleMatrix(n, m);
                for (int j = 0; j < n; j++) {
                    Gd.putRow(j, v.add(u.get(j)));
                }

                Gd.addi(1e-16);

                G = G.mul(K).div(Gd);
            }

            double err = G.distance2(G_prev);

            if (err < stopThr)
                break;
        }

        return G;
    }
}

class Sinkhorn {
    private DoubleMatrix a, b;
    private int numIterMax, n, m;
    private double reg, stopThr;

    public Sinkhorn(DoubleMatrix a_, DoubleMatrix b_, double reg_) {
        a = a_;
        b = b_;
        n = a.length;
        m = b.length;
        reg = reg_;
        numIterMax = 1000;
        stopThr = 1e-9;
    }

    public DoubleMatrix sinkhorn_dist(DoubleMatrix M) {
        DoubleMatrix K = new DoubleMatrix(n, m);

        DoubleMatrix u = DoubleMatrix.ones(n).div(n);
        DoubleMatrix v = DoubleMatrix.ones(m).div(m);

        K = M.div(-reg);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                K.put(i, j, Math.exp(K.get(i, j)));
            }
        }

        DoubleMatrix Kp = K.divColumnVector(a);

        double err = 1;
        mainloop: for (int ii = 0; ii < numIterMax; ii++) {
            DoubleMatrix uprev = u;
            DoubleMatrix vprev = v;
            DoubleMatrix  KtransposeU = K.transpose().mmul(u).rowSums();
            v = b.div(KtransposeU);
            u = DoubleMatrix.ones(n).div(Kp.mmul(v).rowSums());

            double[] u_as_array = u.toArray();
            double[] v_as_array = v.toArray();

            if (Arrays.stream(KtransposeU.toArray()).anyMatch(x -> x == 0) || Arrays.stream(u_as_array).anyMatch(Double::isNaN) || 
                Arrays.stream(v_as_array).anyMatch(Double::isNaN)          || Arrays.stream(u_as_array).anyMatch(Double::isInfinite) || 
                Arrays.stream(v_as_array).anyMatch(Double::isInfinite)) {
                // Reached the machine precision
                u = uprev;
                v = vprev;
                break mainloop;
            }

            if (ii % 10 == 0) {
                // Checking error for every 10 loops
                DoubleMatrix tmp2 = new DoubleMatrix(m);
                for (int j = 0; j < m; j++) {
                    for (int i = 0; i < n; i++) {
                        tmp2.put(j, tmp2.get(j) + u.get(i) * K.get(i, j) * v.get(j));
                    }
                }

                err = tmp2.sub(b).norm1();
                if (err < stopThr)
                    break mainloop;
            }
        }

        return K.mulColumnVector(u).mulRowVector(v);
    }
}

public class GOW {

    private DoubleMatrix a, b, D;
    private DoubleMatrix F;
    private double epsilon, LAMBDA1, LAMBDA2;
    private int n, m, num_function, num_FW_Iter, maxIterNum;
    private String stoppingCriterion;

    public GOW(double[] a_, double[] b_, double[][] D_, int maxIterNum_, double LAMBDA1_, double LAMBDA2_, double epsilon_, int num_FW_Iter_, String stoppingCriterion_){

        a = new DoubleMatrix(a_);
        b = new DoubleMatrix(b_);
        D = new DoubleMatrix(D_);
        maxIterNum = maxIterNum_;
        LAMBDA1 = LAMBDA1_;
        LAMBDA2 = LAMBDA2_;
        epsilon = epsilon_;
        num_FW_Iter = num_FW_Iter_;
        stoppingCriterion = stoppingCriterion_;

        n = a.length;
        m = b.length;
    }

    public static DoubleMatrix compute_F_matrix(Monotonic_Function[] function_list, int m, int num_function) {
        DoubleMatrix F_matrix = new DoubleMatrix(m, num_function);

        for (int j = 0; j < m; j++)
            for (int k = 0; k < num_function; k++)
                try {
                    F_matrix.put(j, k, Monotonic_Function.compute_f(function_list[k], j));
                } catch (InvalidAlgorithmParameterException e) {
                    e.printStackTrace();
                }

        return F_matrix;
    }

    public void setF(DoubleMatrix F_) {
        F = F_;
        num_function = F.getColumns();
    }

    private DoubleMatrix choose_initial_w(DoubleMatrix Y, DoubleMatrix V, int num_function){
        DoubleMatrix initial_w = new DoubleMatrix(num_function);
        double minS = Double.POSITIVE_INFINITY;
        int minIndex = 0;

        for (int i = 0; i < num_function; i++) {
        
            double sum_squared = Y.squaredDistance(V.getColumn(i));

            if (sum_squared < minS) {
                minIndex = i;
                minS = sum_squared;
            }
        }

        initial_w.put(minIndex, 1); 

        return initial_w;
    }

    public DoubleMatrix computeNewCost(DoubleMatrix w) {
        DoubleMatrix D_new = new DoubleMatrix(n, m);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) { 
                DoubleMatrix rowF = F.getRow(j);
                double increase_part = Math.pow((i - w.dot(rowF)) / n, 2);
                D_new.put(i, j, D.get(i, j) + LAMBDA1 * increase_part);
            }

        return D_new;
    }

    public DoubleMatrix computeNewCost2(DoubleMatrix w) {
        DoubleMatrix D_new = new DoubleMatrix(n, m);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) { 
                DoubleMatrix rowF = F.getRow(j);
                double increase_part = Math.pow(1.0*i/n - w.dot(rowF)/m, 2);
                D_new.put(i, j, D.get(i, j) + LAMBDA1 * increase_part);
            }

        return D_new;
    }

    public DoubleMatrix optimize_w(DoubleMatrix T) {
        int index_Y = 0;
        DoubleMatrix Y = new DoubleMatrix(n*m), V = new DoubleMatrix(n*m, num_function);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double temp = Math.sqrt(T.get(i, j));
                Y.put(index_Y, temp * i);

                for (int k = 0; k < num_function; k++) {
                    V.put(index_Y, k, temp * F.get(j, k));
                }

                index_Y += 1;
            }
        }

        DoubleMatrix w = choose_initial_w(Y, V, num_function);

        // Frank-Wolfe algorithm
        for (int i = 0; i < num_FW_Iter; i++) {
            DoubleMatrix jacobian = (V.transpose().mmul(Y.sub(V.mmul(w)))).mul(-2);
            int minIndex = jacobian.argmin();
            DoubleMatrix s = DoubleMatrix.zeros(num_function);
            s.put(minIndex, 1);
            double stepSize = 2.0 / (i + 2);
            w = w.add(s.sub(w).mul(stepSize));
        }

        return w;
    }

    public double GOW_distance_unbalanced() throws InvalidAlgorithmParameterException {
        int iterCount = 0;
        MM_Unbalanced mm_unbalanced = new MM_Unbalanced(a, b, LAMBDA2, "kl");
        DoubleMatrix D_new = D, T = new DoubleMatrix(), w = new DoubleMatrix(), w_old = new DoubleMatrix();
        double diff;

        whileloop: while (iterCount < maxIterNum) {
            iterCount++;

            T = mm_unbalanced.unbalanced_ot_dist(D_new);
            w = optimize_w(T);

            // Checking stoppping criterion
            switch(stoppingCriterion) {
                case "w_slope":
                    if (iterCount != 1) {
                        diff = w.sub(w_old).normmax();

                        int cp = Double.compare(diff, epsilon);

                        if (cp <= 0) {
                            break whileloop;
                        }
                    }

                    break;
                default:
                    throw new InvalidAlgorithmParameterException("Invalid stopping criterion. Valid criteria: \"w_slope\"");

            }

            D_new = computeNewCost(w);

            w_old = w;
            
        }

        return T.dot(D);
    }

    public double GOW_distance_sinkhorn() throws InvalidAlgorithmParameterException {
        int iterCount = 0;
        Sinkhorn sinkhorn = new Sinkhorn(a, b, 1.0/LAMBDA2);
        DoubleMatrix D_new = D, T = new DoubleMatrix(), w = new DoubleMatrix(), w_old = new DoubleMatrix();
        double diff;

        whileloop: while (iterCount < maxIterNum) {
            iterCount++;

            T = sinkhorn.sinkhorn_dist(D_new);
            w = optimize_w(T);

            // Checking stoppping criterion
            switch(stoppingCriterion) {
                case "w_slope":
                    if (iterCount != 1) {
                        diff = w.sub(w_old).normmax();

                        int cp = Double.compare(diff, epsilon);

                        if (cp <= 0) {
                            break whileloop;
                        }
                    }

                    break;
                default:
                    throw new InvalidAlgorithmParameterException("Invalid stopping criterion. Valid criteria: \"w_slope\"");

            }

            D_new = computeNewCost2(w);

            w_old = w;
            
        }

        return T.dot(D);
    }

    public double GOW_distance_sinkhorn_autoscale() throws InvalidAlgorithmParameterException {
        int iterCount = 0;
        Sinkhorn sinkhorn = new Sinkhorn(a, b, 1.0/LAMBDA2);
        DoubleMatrix D_new, T = new DoubleMatrix(), w = new DoubleMatrix(), w_old = new DoubleMatrix();
        double diff;

        int i_scale = n - 1;
        int j_scale = m - 1;

        Monotonic_Function func1 = new Monotonic_Function("polynomial_with_degree", new double[] {1.0*i_scale, 1.0/j_scale, 0.05});
        Monotonic_Function func2 = new Monotonic_Function("polynomial_with_degree", new double[] {1.0*i_scale, 1.0/j_scale, 0.28});
        Monotonic_Function func3 = new Monotonic_Function("polynomial", new double[] {1.0*i_scale/j_scale, 0});
        Monotonic_Function func4 = new Monotonic_Function("polynomial_with_degree", new double[] {1.0*i_scale, 1.0/j_scale, 3.2});
        Monotonic_Function func5 = new Monotonic_Function("polynomial_with_degree", new double[] {1.0*i_scale, 1.0/j_scale, 20});

        Monotonic_Function[] function_list = {func1, func2, func3, func4, func5};

        DoubleMatrix F_matrix = GOW.compute_F_matrix(function_list, m, function_list.length);
        setF(F_matrix);
        
        DoubleMatrix w_0 = new DoubleMatrix(function_list.length);
        w_0.put(2, 1);
        D_new = computeNewCost(w_0);

        whileloop: while (iterCount < maxIterNum) {
            iterCount++;

            T = sinkhorn.sinkhorn_dist(D_new);
            w = optimize_w(T);

            // Checking stoppping criterion
            switch(stoppingCriterion) {
                case "w_slope":
                    if (iterCount != 1) {
                        diff = w.sub(w_old).normmax();

                        int cp = Double.compare(diff, epsilon);

                        if (cp <= 0) {
                            break whileloop;
                        }
                    }

                    break;
                default:
                    throw new InvalidAlgorithmParameterException("Invalid stopping criterion. Valid criteria: \"w_slope\"");

            }

            D_new = computeNewCost(w);

            w_old = w;
            
        }

        return T.dot(D);
    }

    public static void main(String[] args) throws InvalidAlgorithmParameterException {
    }
}