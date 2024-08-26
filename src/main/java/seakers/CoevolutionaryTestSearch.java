package seakers;

import org.moeaframework.algorithm.AbstractEvolutionaryAlgorithm;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.comparator.DominanceComparator;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import seakers.datastructure.ConstraintWeightsData;
import seakers.problems.CoevolutionaryProblem;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.Callable;

public class CoevolutionaryTestSearch implements Callable<Algorithm> {

    private Algorithm coevolutionaryAlgorithm;
    private int numberOfOptimizationObjectives;
    private int maxCoevolutionEvaluations;
    private Result result;
    private DominanceComparator coevolutionComparator;
    private boolean cMOEA;
    private final int runNumber;
    private final boolean evolvePopulation;
    private final boolean integerWeights;
    private final boolean weightOfWeights;
    private final boolean periodicZeroInjection;
    private final String moeaChoice;
    private final String problemChoice;

    public CoevolutionaryTestSearch(Algorithm algorithm, int maxCoevolutionEvaluations, int numberOfOptimizationObjectives, Result result, DominanceComparator coevolutionComparator, boolean cMOEA, boolean evolvePopulation, boolean integerWeights, boolean weightOfWeights, boolean periodicZeroInjection, int runNumber, String moeaChoice, String problemChoice) {
        this.coevolutionaryAlgorithm = algorithm;
        this.maxCoevolutionEvaluations = maxCoevolutionEvaluations;
        this.numberOfOptimizationObjectives = numberOfOptimizationObjectives;
        this.result = result;
        this.coevolutionComparator = coevolutionComparator;
        this.cMOEA = cMOEA;
        this.runNumber = runNumber;
        this.evolvePopulation = evolvePopulation;
        this.integerWeights = integerWeights;
        this.weightOfWeights = weightOfWeights;
        this.periodicZeroInjection = periodicZeroInjection;
        this.moeaChoice = moeaChoice;
        this.problemChoice = problemChoice;
    }

    @Override
    public Algorithm call() throws Exception {
        // Run Outer MOEA (evolving population of heuristic weights)
        System.out.println("Starting Coevolutionary Algorithm");

        int numberOfWeightsObjectives = 1;
        if (((CoevolutionaryProblem) coevolutionaryAlgorithm.getProblem()).getCMOEA()) {
            numberOfWeightsObjectives = numberOfOptimizationObjectives;
        }

        long startTime = System.currentTimeMillis();
        coevolutionaryAlgorithm.step();

        System.out.println("NFE = 0");

        while (!coevolutionaryAlgorithm.isTerminated() && (coevolutionaryAlgorithm.getNumberOfEvaluations() < maxCoevolutionEvaluations)) {
            if (periodicZeroInjection) {
                Solution zeroSolution = ((CoevolutionaryProblem) coevolutionaryAlgorithm.getProblem()).newSolution();
                for (int i = 0; i < zeroSolution.getNumberOfVariables(); i++) {
                    if (zeroSolution.getVariable(i) instanceof RealVariable) {
                        RealVariable zeroVar = null;
                        if (weightOfWeights) {
                            if (i == zeroSolution.getNumberOfVariables()-1) { // last variable in weightOfWeights run is the weight of weights
                                zeroVar = new RealVariable(0, ((RealVariable) zeroSolution.getVariable(i)).getLowerBound(), ((RealVariable) zeroSolution.getVariable(i)).getUpperBound());
                            }
                        } else {
                            if (integerWeights) {
                                zeroVar = new RealVariable(0, ((RealVariable) zeroSolution.getVariable(i)).getLowerBound(), ((RealVariable) zeroSolution.getVariable(i)).getUpperBound());
                            } else {
                                zeroVar = new RealVariable(-3, ((RealVariable) zeroSolution.getVariable(i)).getLowerBound(), ((RealVariable) zeroSolution.getVariable(i)).getUpperBound());
                            }
                        }
                        zeroSolution.setVariable(i, zeroVar);
                    }
                }
                BinaryVariable evolvePopulationVariable = new BinaryVariable(1);
                if (evolvePopulation) {
                    EncodingUtils.setBoolean(evolvePopulationVariable, true);
                } else {
                    EncodingUtils.setBoolean(evolvePopulationVariable, false);
                }
                zeroSolution.setVariable(zeroSolution.getNumberOfVariables()-1, evolvePopulationVariable);
                //coevolutionaryAlgorithm.getProblem().evaluate(zeroSolution);

                // Remove least fit solution and add zero solution
                Population currentPopulation = ((AbstractEvolutionaryAlgorithm) coevolutionaryAlgorithm).getPopulation();
                Solution lowestFitnessSolution = currentPopulation.get(0);
                for (int i = 1; i < currentPopulation.size(); i++) {
                    if (coevolutionComparator.compare(lowestFitnessSolution, currentPopulation.get(i)) == -1) { // high objective means low fitness (solution objectives must be minimized)
                        lowestFitnessSolution = currentPopulation.get(i);
                    }
                }
                ((AbstractEvolutionaryAlgorithm) coevolutionaryAlgorithm).getPopulation().remove(lowestFitnessSolution);
                ((AbstractEvolutionaryAlgorithm) coevolutionaryAlgorithm).getPopulation().add(zeroSolution);
            }
            coevolutionaryAlgorithm.step();
        }

        coevolutionaryAlgorithm.terminate();
        long endTime = System.currentTimeMillis();

        System.out.println("Coevolutionary Algorithm completed in " + ((endTime - startTime)/1000.0) + "s");

        ArrayList<ConstraintWeightsData> allWeights = ((CoevolutionaryProblem) coevolutionaryAlgorithm.getProblem()).getAllWeights();

        Population finalPopulation = ((AbstractEvolutionaryAlgorithm) coevolutionaryAlgorithm).getPopulation();
        result.saveFinalPopulationOrArchive(finalPopulation, "run " + runNumber + File.separator + problemChoice + moeaChoice + "coevolutionary_algorithm_constraint_weights_finalpop.csv", numberOfWeightsObjectives, ((CoevolutionaryProblem) coevolutionaryAlgorithm.getProblem()).getOptimizationProblem().getNumberOfConstraints());

        if (cMOEA) {
            NondominatedPopulation finalArchive = ((AbstractEvolutionaryAlgorithm) coevolutionaryAlgorithm).getArchive();
            result.saveFinalPopulationOrArchive(finalArchive, "run " + runNumber + File.separator + problemChoice + moeaChoice + "coevolutionary_algorithm_constraint_weights_finalarchive.csv", numberOfWeightsObjectives, ((CoevolutionaryProblem) coevolutionaryAlgorithm.getProblem()).getOptimizationProblem().getNumberOfConstraints());
        }

        result.saveConstraintWeights(allWeights, numberOfWeightsObjectives, ((CoevolutionaryProblem) coevolutionaryAlgorithm.getProblem()).getOptimizationProblem().getNumberOfConstraints(), "run " + runNumber + File.separator + problemChoice + moeaChoice + "coevolutionary_algorithm_constraint_weights.csv", true);

        return coevolutionaryAlgorithm;
    }
}
