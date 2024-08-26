package seakers.operators;

import org.moeaframework.core.PRNG;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variable;
import org.moeaframework.core.Variation;
import org.moeaframework.core.variable.EncodingUtils;

public class UniformIntegerWeightsMutation implements Variation {

    private final double probability;

    public UniformIntegerWeightsMutation(double probability) {
        this.probability = probability;
    }

    public double getProbability() {return this.probability; }

    @Override
    public int getArity() {
        return 1;
    }

    @Override
    public Solution[] evolve(Solution[] parents) {
        Solution result = parents[0].copy();

        for(int i = 0; i < result.getNumberOfVariables()-1; ++i) { // only the decisions corresponding to the heuristic weights
            Variable variable = result.getVariable(i);
            if (PRNG.nextDouble() <= this.probability) {
                EncodingUtils.setInt(variable, PRNG.nextInt(11));
                result.setVariable(i, variable);
            }
        }

        return new Solution[]{result};
    }
}
