package seakers.utils;

import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Solution;
import org.moeaframework.core.indicator.Hypervolume;
import org.moeaframework.problem.AbstractProblem;

public class ObjectiveHypervolume extends Hypervolume {

    /**
     * Class to compute hypervolume of objective-based Pareto Front (extends MOEAFramework's Hypervolume class
     * which does not consider constraint violating solutions
     */

    public ObjectiveHypervolume(AbstractProblem problem, double[] minimum, double[] maximum) {
        super(problem, minimum, maximum);
    }

    public double computeObjectiveHypervolume(NondominatedPopulation population) {
        // Create duplicate population with constraints = 0 so that Hypervolume's evaluate method considers them
        NondominatedPopulation noConstraintsPopulation = new NondominatedPopulation();
        for (Solution solution : population) {
            Solution noConstraintsSolution = solution.deepCopy();
            for (int i = 0; i < solution.getNumberOfConstraints(); i++) {
                noConstraintsSolution.setConstraint(i, 0.0);
            }
            noConstraintsPopulation.add(noConstraintsSolution);
        }

        // Compute and return Hypervolume
        return super.evaluate(noConstraintsPopulation);
    }
}
