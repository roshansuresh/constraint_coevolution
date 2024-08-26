package seakers.operators;

import org.moeaframework.core.PRNG;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variable;
import org.moeaframework.core.Variation;

public class UniformWeightsCrossover implements Variation {

    private final double probability;

    public UniformWeightsCrossover(double probability) {
        this.probability = probability;
    }

    @Override
    public int getArity() {return 2;}

    @Override
    public Solution[] evolve(Solution[] parents) {
        Solution child1 = parents[0].copy();
        Solution child2 = parents[1].copy();
        if (PRNG.nextDouble() <= probability) {
            for (int i = 0; i < child1.getNumberOfVariables()-1; ++i) { // only the decisions corresponding to the heuristic weights
                if (PRNG.nextBoolean()) {
                    Variable temp = child1.getVariable(i);
                    child1.setVariable(i, child2.getVariable(i));
                    child2.setVariable(i, temp);
                }
            }
        }

        return new Solution[]{child1, child2};
    }
}
