import org.jblas.DoubleMatrix;

public class AWSWD {

    private double w_1, w_2, w_3;
    private int maxLoop, subsequence_len;
    private double midpoint, steepness;
    private double LAMBDA, epsilon;
    private DoubleMatrix seq1, seq2;
    private int len1, len2, num_dimension;

    public AWSWD(double[][] seq1_, double[][] seq2_, double midpoint_, int maxLoop_, double LAMBDA_, int subsequence_len_, double steepness_, double epsilon_){
        seq1 = new DoubleMatrix(seq1_);
        seq2 = new DoubleMatrix(seq2_);
        len1 = seq1.getRows();
        len2 = seq2.getRows();
        midpoint = midpoint_;
        maxLoop = maxLoop_;
        LAMBDA = LAMBDA_;
        subsequence_len = subsequence_len_;
        steepness = steepness_;
        epsilon = epsilon_;
        num_dimension = seq1.getColumns();
    }

    public DoubleMatrix compute_ground_cost_matrix_1() {
        DoubleMatrix D = new DoubleMatrix(len1, len2);

        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                D.put(i, j, seq1.getRow(i).distance2(seq2.getRow(j)));
            }
        }

        return D;
    }

    public DoubleMatrix differential(DoubleMatrix seq, int element_index) {

        DoubleMatrix temp1, temp2;

        if (element_index == 0) {
            temp1 = seq.getRow(element_index).sub(DoubleMatrix.zeros(num_dimension));
            temp2 = seq.getRow(element_index + 1).sub(DoubleMatrix.zeros(num_dimension)).mul(0.5);
        } else if (element_index == seq.getRows() - 1) {
            temp1 = seq.getRow(element_index).sub(seq.getRow(element_index - 1));
            temp2 = DoubleMatrix.zeros(num_dimension).sub(seq.getRow(element_index - 1)).mul(0.5);
        } else {
            temp1 = seq.getRow(element_index).sub(seq.getRow(element_index - 1));
            temp2 = seq.getRow(element_index + 1).sub(seq.getRow(element_index - 1)).mul(0.5);
        }

        return  temp1.add(temp2).mul(0.5);
    }

    public DoubleMatrix compute_ground_cost_matrix_2() {
        DoubleMatrix D = new DoubleMatrix(len1, len2);

        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                for (int k = -subsequence_len; k <= subsequence_len; k++) {
                    int new_i = i + k;
                    int new_j = j + k;
                    if ((0 <= new_i) && (new_i < len1) && (0 <= new_j) && (new_j < len2)) {
                        D.put(i, j, D.get(i, j) + differential(seq1, new_i).distance2(differential(seq2, new_j)));
                    }
                }
            }
        }
        
        return D;
    }

    public DoubleMatrix compute_ground_cost_matrix_3() {

        DoubleMatrix D = new DoubleMatrix(len1, len2);

        for (int i = 0; i < len1; i++) {
            for (int j = 0; j < len2; j++) {
                D.put(i, j, 1.0/(1.0 + Math.exp(-steepness * (Math.abs(i - j) - midpoint))));
            }
        }

        return D;
    }

    public boolean w_reached_maximum_value(double new_w1, double new_w2, double new_w3) {
        if ((new_w1 >= 1e3) || (new_w2 >= 1e3) || (new_w3 >= 1e3) || Double.isNaN(new_w1) || Double.isNaN(new_w2) || Double.isNaN(new_w3))
            return true;

        return false;
    }

    public boolean w_converged(double new_w1, double new_w2, double new_w3, double old_w1, double old_w2, double old_w3) {
        double diff = Math.max(Math.max(Math.abs(new_w1 - old_w1), Math.abs(new_w2 - old_w2)), Math.abs(new_w3 - old_w3));
        
        if (diff <= epsilon)
            return true;

        return false;
    }

    public double update_weight(DoubleMatrix new_T, DoubleMatrix distance_matrix) {
        return 0.5 / Math.sqrt(new_T.dot(distance_matrix));
    }

    public double auto_weighted_sequential_wasserstein_distance() {
        w_1 = 1.0/3;
        w_2 = 1.0/3;
        w_3 = 1.0/3;

        DoubleMatrix a = DoubleMatrix.ones(len1).div(len1);
        DoubleMatrix b = DoubleMatrix.ones(len2).div(len2);

        DoubleMatrix D_1 = compute_ground_cost_matrix_1();
        DoubleMatrix D_2 = compute_ground_cost_matrix_2();
        DoubleMatrix D_3 = compute_ground_cost_matrix_3();

        Sinkhorn sinkhorn = new Sinkhorn(a, b, 1.0/LAMBDA);
        DoubleMatrix T = new DoubleMatrix(), D_sum = new DoubleMatrix();

        int loop = 0;

        while (true) {
            loop++;

            D_sum = D_1.mul(w_1).add(D_2.mul(w_2)).add(D_3.mul(w_3));
            T = sinkhorn.sinkhorn_dist(D_sum);

            double w_1_new = update_weight(T, D_1);
            double w_2_new = update_weight(T, D_2);
            double w_3_new = update_weight(T, D_3);

            if ((w_converged(w_1_new, w_2_new, w_3_new, w_1, w_2, w_3)) || (loop == maxLoop))
                break;

            w_1 = w_1_new;
            w_2 = w_2_new;
            w_3 = w_3_new;
        }

        return T.dot(D_sum);
    }

    public double normalized_auto_weighted_sequential_wasserstein_distance() {
        w_1 = 1.0/3;
        w_2 = 1.0/3;
        w_3 = 1.0/3;

        DoubleMatrix a = DoubleMatrix.ones(len1).div(len1);
        DoubleMatrix b = DoubleMatrix.ones(len2).div(len2);

        DoubleMatrix D_1 = new DoubleMatrix(len1, len2); 
        DoubleMatrix D_2 = new DoubleMatrix(len1, len2);
        DoubleMatrix D_3 = new DoubleMatrix(len1, len2);

        DoubleMatrix seq1_diff = new DoubleMatrix(len1, num_dimension);
        DoubleMatrix seq2_diff = new DoubleMatrix(len2, num_dimension);
        DoubleMatrix diff_matrix = new DoubleMatrix(len1, len2);

        for (int i = 0; i < len1; i++)
            seq1_diff.putRow(i, differential(seq1, i));

        for (int i = 0; i < len2; i++)
            seq2_diff.putRow(i, differential(seq2, i));

        for (int i = 0; i < len1; i++)
            for (int j = 0; j < len2; j++)
                diff_matrix.put(i, j, seq1_diff.getRow(i).distance2(seq2_diff.getRow(j)));

        for (int i = 0; i < len1; i++) {
            DoubleMatrix temp = seq1.getRow(i);
            for (int j = 0; j < len2; j++) {
                // Computing D1
                D_1.put(i, j, temp.distance2(seq2.getRow(j)));

                // Computing D2
                if (i == 0 || j == 0) {
                    double sum = 0.0;
                    for (int k = -subsequence_len; k <= subsequence_len; k++) {
                        int new_i = i + k;
                        int new_j = j + k;

                        if ((0 <= new_i) && (new_i < len1) && (0 <= new_j) && (new_j < len2)) {
                            sum += diff_matrix.get(new_i, new_j);
                        }
                    }

                    D_2.put(i, j, sum);
                } else {
                    double temp1 = 0.0, temp2 = 0.0;
                    int new_i = i - subsequence_len - 1, new_j = j - subsequence_len - 1;
                    
                    if (new_i >= 0 && new_j >= 0)
                        temp1 = diff_matrix.get(new_i, new_j);

                    new_i = i + subsequence_len;
                    new_j = j + subsequence_len;    

                    if (new_i < len1 && new_j < len2)
                        temp2 = diff_matrix.get(new_i, new_j);
                    
                    D_2.put(i, j, D_2.get(i-1, j-1) - temp1 + temp2);
                }

                // Computing D_3
                D_3.put(i, j, 1.0/(1.0 + Math.exp(-steepness * (Math.abs(i - j) - midpoint))));
            }
        }

        D_1.divi(D_1.max());
        D_2.divi(D_2.max());
        D_3.divi(D_3.max());

        Sinkhorn sinkhorn = new Sinkhorn(a, b, 1.0/LAMBDA);
        DoubleMatrix T = new DoubleMatrix(), D_sum = new DoubleMatrix();

        int loop = 0;

        while (true) {
            loop++;

            D_sum = D_1.mul(w_1).add(D_2.mul(w_2)).add(D_3.mul(w_3));
            T = sinkhorn.sinkhorn_dist(D_sum);

            double w_1_new = update_weight(T, D_1);
            double w_2_new = update_weight(T, D_2);
            double w_3_new = update_weight(T, D_3);

            if (w_converged(w_1_new, w_2_new, w_3_new, w_1, w_2, w_3) || w_reached_maximum_value(w_1_new, w_2_new, w_3_new) || (loop == maxLoop))
                break;

            w_1 = w_1_new;
            w_2 = w_2_new;
            w_3 = w_3_new;
        }

        return T.dot(D_sum);
    }

    public double faster_auto_weighted_sequential_wasserstein_distance() {
        w_1 = 1.0/3;
        w_2 = 1.0/3;
        w_3 = 1.0/3;

        DoubleMatrix a = DoubleMatrix.ones(len1).div(len1);
        DoubleMatrix b = DoubleMatrix.ones(len2).div(len2);

        DoubleMatrix D_1 = new DoubleMatrix(len1, len2); 
        DoubleMatrix D_2 = new DoubleMatrix(len1, len2);
        DoubleMatrix D_3 = new DoubleMatrix(len1, len2);

        DoubleMatrix seq1_diff = new DoubleMatrix(len1, num_dimension);
        DoubleMatrix seq2_diff = new DoubleMatrix(len2, num_dimension);
        DoubleMatrix diff_matrix = new DoubleMatrix(len1, len2);

        for (int i = 0; i < len1; i++)
            seq1_diff.putRow(i, differential(seq1, i));

        for (int i = 0; i < len2; i++)
            seq2_diff.putRow(i, differential(seq2, i));

        for (int i = 0; i < len1; i++)
            for (int j = 0; j < len2; j++)
                diff_matrix.put(i, j, seq1_diff.getRow(i).distance2(seq2_diff.getRow(j)));

        for (int i = 0; i < len1; i++) {
            DoubleMatrix temp = seq1.getRow(i);
            for (int j = 0; j < len2; j++) {
                // Computing D1
                D_1.put(i, j, temp.distance2(seq2.getRow(j)));

                // Computing D2
                if (i == 0 || j == 0) {
                    double sum = 0.0;
                    for (int k = -subsequence_len; k <= subsequence_len; k++) {
                        int new_i = i + k;
                        int new_j = j + k;

                        if ((0 <= new_i) && (new_i < len1) && (0 <= new_j) && (new_j < len2)) {
                            sum += diff_matrix.get(new_i, new_j);
                        }
                    }

                    D_2.put(i, j, sum);
                } else {
                    double temp1 = 0.0, temp2 = 0.0;
                    int new_i = i - subsequence_len - 1, new_j = j - subsequence_len - 1;
                    
                    if (new_i >= 0 && new_j >= 0)
                        temp1 = diff_matrix.get(new_i, new_j);

                    new_i = i + subsequence_len;
                    new_j = j + subsequence_len;    

                    if (new_i < len1 && new_j < len2)
                        temp2 = diff_matrix.get(new_i, new_j);
                    
                    D_2.put(i, j, D_2.get(i-1, j-1) - temp1 + temp2);
                }

                // Computing D_3
                D_3.put(i, j, 1.0/(1.0 + Math.exp(-steepness * (Math.abs(i - j) - midpoint))));
            }
        }

        Sinkhorn sinkhorn = new Sinkhorn(a, b, 1.0/LAMBDA);
        DoubleMatrix T = new DoubleMatrix(), D_sum = new DoubleMatrix();
        
        int loop = 0;
        while (true) {
            loop++;

            D_sum = D_1.mul(w_1).add(D_2.mul(w_2)).add(D_3.mul(w_3));
            T = sinkhorn.sinkhorn_dist(D_sum);

            double w_1_new = update_weight(T, D_1);
            double w_2_new = update_weight(T, D_2);
            double w_3_new = update_weight(T, D_3);

            if ((w_converged(w_1_new, w_2_new, w_3_new, w_1, w_2, w_3)) || (loop == maxLoop))
                break;

            w_1 = w_1_new;
            w_2 = w_2_new;
            w_3 = w_3_new;
        }

        return T.dot(D_sum);
    }
}