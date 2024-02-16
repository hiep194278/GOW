import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import java.util.Arrays;

class Sinkhorn {
    private DoubleMatrix a, b;
    private int numIterMax, n, m;
    private double reg, stopThr;

    public Sinkhorn(DoubleMatrix a_, DoubleMatrix b_, double reg_) {
        a = a_;
        b = b_;
        n = a.getRows();
        m = b.getRows();
        reg = reg_;
        numIterMax = 1000;
        stopThr = 1e-9;
    }

    public DoubleMatrix sinkhorn_dist(DoubleMatrix M) {
        DoubleMatrix K = new DoubleMatrix(n, m);

        DoubleMatrix u = DoubleMatrix.ones(n).div(n);
        DoubleMatrix v = DoubleMatrix.ones(m).div(m);

        K = M.div(-reg);
        MatrixFunctions.expi(K);

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

