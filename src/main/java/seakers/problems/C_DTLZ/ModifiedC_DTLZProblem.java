package seakers.problems.C_DTLZ;

import org.moeaframework.core.Solution;
import org.moeaframework.problem.AbstractProblem;
import seakers.problems.AbstractInternalProblem;

import java.util.Arrays;

public class ModifiedC_DTLZProblem extends AbstractProblem implements AbstractInternalProblem {

    private AbstractProblem cdltzProblem;
    private double[] constraintWeights;
    private boolean integerWeights;
    private final double[] objectiveNorms;
    private final double[] constraintNorms;
    //private final double[] objectiveNorms;

    public ModifiedC_DTLZProblem(AbstractProblem cdltzProblem, int numberOfConstraints, boolean integerWeights, double[] objectiveNorms, double[] constraintNorms) {
        super(cdltzProblem.getNumberOfVariables(), cdltzProblem.getNumberOfObjectives(), cdltzProblem.getNumberOfConstraints());
        this.cdltzProblem = cdltzProblem;
        this.constraintWeights = new double[numberOfConstraints];
        this.integerWeights = integerWeights;
        this.objectiveNorms = objectiveNorms;
        this.constraintNorms = constraintNorms;
        //this.objectiveNorms = objectiveNorms;
    }

    @Override
    public void evaluate(Solution solution) {
        // First evaluate solution using underlying C_DTLZ problem to obtain non-penalized objectives and constraints
        cdltzProblem.evaluate(solution);
        double norm = 1.0;
        if (integerWeights) { // If real constraint weights are used, then don't normalize
            norm = 10.0;
            double sum = Arrays.stream(constraintWeights).sum();
            if (sum == 0.0) {
                norm = 1e-5;
            }
        }


        for (int i = 0; i < this.cdltzProblem.getNumberOfConstraints(); i++) {
            double constraint = 0.0;
            if (!this.cdltzProblem.getName().equals("C2_DTLZ2")) { // The constraints in all C-DTLZ problems except C2-DTLZ2 are of the form c >= 0
                constraint = -solution.getConstraint(i)/constraintNorms[i];
            } else {
                constraint = solution.getConstraint(i)/constraintNorms[i];
            }
            solution.setConstraint(i, constraint);
        }

        for (int i = 0; i < solution.getNumberOfObjectives(); i++) {
            solution.setAttribute("TrueObjective" + (i+1), solution.getObjective(i)/objectiveNorms[i]);
        }

        // Add constraint violation penalties to objectives and set modified objectives to solution
        double totalConstraintPenalty = 0.0;
        for (int i = 0; i < numberOfConstraints; i++) {
            totalConstraintPenalty += (constraintWeights[i]/norm)*(solution.getConstraint(i)/numberOfConstraints);
        }

        for (int i = 0; i < solution.getNumberOfObjectives(); i++) {
            //double penalizedObjective = (solution.getObjective(i)/objectiveNorms[i]) + totalConstraintPenalty;
            double penalizedObjective = (solution.getObjective(i)/objectiveNorms[i]) + totalConstraintPenalty;
            solution.setObjective(i, penalizedObjective);
        }
        //if (!(boolean) solution.getAttribute("AlreadyEvaluated")) {

            //solution.setAttribute("AlreadyEvaluated", true);
        //}
    }

    @Override
    public Solution newSolution() {
        Solution solution = this.cdltzProblem.newSolution();
        for (int j = 0; j < solution.getNumberOfVariables(); j++) {
            solution.getVariable(j).randomize();
        }
        //solution.setAttribute("AlreadyEvaluated", false);
        return solution;
    }

    @Override
    public void setConstraintWeights(double[] constraintWeights) {
        this.constraintWeights = constraintWeights;
    }
}
