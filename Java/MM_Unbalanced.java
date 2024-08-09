import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

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
            MatrixFunctions.expi(K);
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
                MatrixFunctions.sqrti(u);
                DoubleMatrix v = b.div(G.columnSums().add(1e-16));
                MatrixFunctions.sqrti(v);
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