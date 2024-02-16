import java.security.InvalidAlgorithmParameterException;

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