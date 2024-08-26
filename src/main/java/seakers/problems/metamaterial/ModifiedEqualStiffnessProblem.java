package seakers.problems.metamaterial;

import com.mathworks.engine.MatlabEngine;
import org.moeaframework.core.Solution;
import seakers.problems.AbstractInternalProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;

import java.util.Arrays;

public class ModifiedEqualStiffnessProblem extends ConstantRadiusTrussProblem2 implements AbstractInternalProblem {

    private double[] constraintWeights;
    private final boolean integerWeights;
    private final String[] constraintNames;

    public ModifiedEqualStiffnessProblem(boolean integerWeights, int numberOfConstraints, int modelSelection, int numberOfVariables, double memberRadius, double sideElementLength, double modulusYoungs, double sideNodeNumber, double nucFac, double targetCRatio, MatlabEngine eng, boolean[][] heuristicEnforcement, String[] constraintNames) {
        super("", modelSelection, numberOfVariables, 0, 0, memberRadius, sideElementLength, modulusYoungs, sideNodeNumber, nucFac, targetCRatio, eng, heuristicEnforcement);
        this.integerWeights = integerWeights;
        this.constraintWeights = new double[numberOfConstraints];
        this.constraintNames = constraintNames;
    }

    @Override
    public void evaluate(Solution solution) {
        // Evaluate design using the parent class
        super.evaluate(solution);
        double norm = 1.0;
        if (integerWeights) { // If real heuristic weights are used, then don't normalize
            norm = 10.0;
            double sum = Arrays.stream(constraintWeights).sum();
            if (sum == 0.0) {
                norm = 1e-5;
            }
        }

        // Add constraint penalization to the objectives (the parent class also computes the constraint violations)
        double totalConstraintPenalty = 0.0;
        for (int i = 0; i < constraintNames.length; i++) {
            //totalConstraintPenalty += (constraintWeights[i]/norm)*((double) solution.getAttribute(constraintNames[i]));
            totalConstraintPenalty += (constraintWeights[i]/norm)*(solution.getConstraint(i)/numberOfConstraints);
        }

        for (int i = 0; i < solution.getNumberOfObjectives(); i++) {
            double penalizedObjective = solution.getObjective(i) + totalConstraintPenalty;
            solution.setObjective(i, penalizedObjective);
        }
    }

    public void setConstraintWeights(double[] newWeights) {
        this.constraintWeights = newWeights;
    }
}
