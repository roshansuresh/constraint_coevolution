package seakers.datastructure;

/**
 * Data structure to store the heuristic weights at each NFE of the main coevolutionary algorithm
 */

public class ConstraintWeightsData {

    public int NFE;
    public double[] constraintWeights;
    public double[] fitness;
    public int numberOfFeasibleSolutions;

    public ConstraintWeightsData(int numberOfConstraints, int numberOfOptimizationObjectives) {
        this.NFE = 0;
        this.fitness = new double[numberOfOptimizationObjectives]; // Number of optimization objectives is used only if CMOEA is true
        this.numberOfFeasibleSolutions = 0;
        this.constraintWeights = new double[numberOfConstraints];
    }

    public int getNFE() {
        return this.NFE;
    }

    public void setNFE(int nfe) {
        this.NFE = nfe;
    }

    public double[] getConstraintWeights() {
        return this.constraintWeights;
    }

    public void setConstraintWeights(double[] heuristicWeights) {
        this.constraintWeights = heuristicWeights;
    }

    public double[] getFitness() {
        return this.fitness;
    }

    public void setFitness(double[] fitness) {
        this.fitness = fitness;
    }

    public int getNumberOfFeasibleSolutions() {return this.numberOfFeasibleSolutions; }

    public void setNumberOfFeasibleSolutions(int numberOfFeasibleDesigns) {this.numberOfFeasibleSolutions = numberOfFeasibleDesigns; }
}
