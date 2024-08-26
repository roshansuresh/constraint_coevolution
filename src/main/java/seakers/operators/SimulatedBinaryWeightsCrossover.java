package seakers.operators;

import org.moeaframework.core.Solution;
import org.moeaframework.core.operator.real.SBX;

public class SimulatedBinaryWeightsCrossover extends SBX {

    public SimulatedBinaryWeightsCrossover(double probability, double distributionIndex) {
        super(probability, distributionIndex);
    }

    @Override
    public Solution[] evolve(Solution[] parents) {

        // Store all but the last binary variable for all solutions into a different solution array (since we don't want the crossover operator to change it)
        Solution[] crossoverParents = new Solution[parents.length];
        for (int i = 0; i < crossoverParents.length; i++) {
            Solution crossoverParent = new Solution(parents[0].getNumberOfVariables()-1, parents[0].getNumberOfObjectives(), parents[0].getNumberOfConstraints());
            for (int j = 0; j < crossoverParent.getNumberOfVariables(); j++) {
                crossoverParent.setVariable(j, parents[i].getVariable(j));
            }
            crossoverParents[i] = crossoverParent;
        }

        // Evolve the created solution array using SBX
        Solution[] SBXChildren = super.evolve(crossoverParents);

        // Copy the evolved variables to children and keep last variable from parents
        Solution[] children = new Solution[parents.length];
        for (int i = 0; i < children.length; i++) {
            Solution child = parents[i].copy();
            for (int j = 0; j < child.getNumberOfVariables()-1; j++) {
                child.setVariable(j, SBXChildren[i].getVariable(j));
            }
            //child.setVariable(child.getNumberOfVariables(), parents[i].getVariable(child.getNumberOfVariables()-1)); // Not required
            children[i] = child;
        }

        return children;
    }
}
