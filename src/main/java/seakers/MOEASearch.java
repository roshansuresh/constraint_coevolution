package seakers;

import org.moeaframework.algorithm.AbstractEvolutionaryAlgorithm;
import org.moeaframework.algorithm.MOEAD;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;

import java.util.ArrayList;
import java.util.concurrent.Callable;

public class MOEASearch implements Callable<Algorithm> {

    private Algorithm algorithm;
    private String saveDirectory;
    private int maximumNFE;
    private int runNumber;
    private final String[] variableNames;
    private final String[] objectiveNames;
    private final String[] constraintNames;
    private final boolean isCDTLZ;
    private final String problemChoice;
    private final int moeaChoice;

    public MOEASearch(Algorithm algorithm, String saveDirectory, int maximumNFE, int runNumber, String[] variableNames, String[] objectiveNames, String[] constraintNames, String problemChoice, boolean isCDTLZ, int moeaChoice) {
        this.algorithm = algorithm;
        this.saveDirectory = saveDirectory;
        this.maximumNFE = maximumNFE;
        this.runNumber = runNumber;
        this.variableNames = variableNames;
        this.objectiveNames = objectiveNames;
        this.constraintNames = constraintNames;
        this.problemChoice = problemChoice;
        this.isCDTLZ = isCDTLZ;
        this.moeaChoice = moeaChoice;
    }

    @Override
    public Algorithm call() throws Exception {
        System.out.println("Starting MOEA Run");

        Result result = new Result(saveDirectory, 0);

        ArrayList<Solution> allSolutions = new ArrayList<>();

        long startTime = System.currentTimeMillis();
        algorithm.step();

        Population initialPopulation;
        if (moeaChoice == 2) {
            initialPopulation = ((MOEAD) algorithm).getResult(); // This is only the current non-dominated population

        } else {
            initialPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
        }
        for (Solution solution : initialPopulation) {
            solution.setAttribute("NFE", 0);
            allSolutions.add(solution);
        }

        while (!algorithm.isTerminated() && (algorithm.getNumberOfEvaluations() < maximumNFE)) {
            algorithm.step();
            if (moeaChoice == 1) { // Epsilon-MOEA
                Population currentPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
                for (int i = 1; i < 3; i++) {
                    Solution currentSolution = currentPopulation.get(currentPopulation.size() - i);
                    currentSolution.setAttribute("NFE", algorithm.getNumberOfEvaluations());
                    allSolutions.add(currentSolution);
                }
            } else { // MOEA-D and IBEA
                Population currentNonDominatedPopulation = algorithm.getResult();
                for (int i = 0; i < currentNonDominatedPopulation.size(); i++) {
                    Solution currentSolution = currentNonDominatedPopulation.get(i);
                    currentSolution.setAttribute("NFE", algorithm.getNumberOfEvaluations());
                    allSolutions.add(currentSolution);
                }
            }
        }

        algorithm.terminate();
        long endTime = System.currentTimeMillis();

        System.out.println("Total Execution Time: " + ((endTime - startTime)/1000.0) + " s");

        String algorithmName = "";
        switch (moeaChoice) {
            case 1: // Epsilon-MOEA
                algorithmName = "EpsilonMOEA_";
                break;
            case 2: // MOEA-D
                algorithmName = "MOEAD_";
                break;
            case 3: // IBEA
                algorithmName = "IBEA_";
                break;
            default:
                System.out.println("Invalid model choice");
                break;
        }

        if (moeaChoice == 1) { // Epsilon-MOEA
            Population finalPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
            NondominatedPopulation finalArchive = ((AbstractEvolutionaryAlgorithm) algorithm).getArchive();

            // save final population
            result.saveInternalPopulationOrArchive( algorithmName + problemChoice + runNumber + "_finalpop.csv", finalPopulation, variableNames, objectiveNames, constraintNames, isCDTLZ, true);

            // save final archive
            result.saveInternalPopulationOrArchive(algorithmName + problemChoice + runNumber + "_finalarchive.csv", finalArchive, variableNames, objectiveNames, constraintNames, isCDTLZ, true);
        } else { // MOEA-D and IBEA
            NondominatedPopulation finalArchive = algorithm.getResult();

            // save final archive
            result.saveInternalPopulationOrArchive(algorithmName + problemChoice + runNumber + "_finalarchive.csv", finalArchive, variableNames, objectiveNames, constraintNames, isCDTLZ, true);
        }

        // save all solutions
        result.saveAllInternalSolutions(algorithmName + problemChoice + runNumber + "_allSolutions.csv", allSolutions, variableNames, objectiveNames, constraintNames, isCDTLZ, true);

        return algorithm;
    }
}
