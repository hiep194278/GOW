import java.security.InvalidAlgorithmParameterException;
import org.jblas.DoubleMatrix;

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
        DoubleMatrix a = new DoubleMatrix(new double[] {1, 2.5, 5});
        DoubleMatrix b = new DoubleMatrix(new double[] {4, 2, 1.4});
        AWSWD awswd = new AWSWD(a.toArray2(), b.toArray2(), 3.0/3, 100, 50, 5, 0.1, 0.01);
        System.out.println(awswd.auto_weighted_sequential_wasserstein_distance());
    }
}