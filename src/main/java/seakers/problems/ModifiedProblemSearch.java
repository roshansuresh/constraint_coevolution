package seakers.problems;

import org.apache.commons.math3.util.Precision;
import org.moeaframework.algorithm.AbstractEvolutionaryAlgorithm;
import org.moeaframework.core.*;
import seakers.Result;

import java.io.File;
import java.util.ArrayList;
import java.util.StringJoiner;

public class ModifiedProblemSearch {

    private final Algorithm algorithm;
    private final Initialization initialization;
    private final String saveDirectory;
    private double[] constraintWeights;
    private final int internalMaxEvaluations;
    private final int currentCoevolutionNFE;
    private final String[] variableNames;
    private final String[] objectiveNames;
    private final String[] constraintNames;
    private final int coevolutionaryRunNumber;
    private final boolean isCDTLZ;
    private final boolean weightOfWeights;
    private final String moeaChoice;
    private final String problemChoice;
    private Population initialPopulation;
    private Population finalPopulation;
    private NondominatedPopulation finalArchive;

    public ModifiedProblemSearch(Algorithm algorithm, Initialization initialization, String saveDirectory, double[] constraintWeights, int internalMaxEvaluations, int currentCoevolutionNFE, String[] variableNames, String[] objectiveNames, String[] constraintNames, boolean isCDTLZ, boolean weightOfWeights, int coevolutionaryRunNumber, String moeaChoice, String problemChoice) {
        this.algorithm = algorithm;
        this.initialization = initialization;
        this.saveDirectory = saveDirectory;
        this.constraintWeights = constraintWeights;
        this.internalMaxEvaluations = internalMaxEvaluations;
        this.currentCoevolutionNFE = currentCoevolutionNFE;
        this.variableNames = variableNames;
        this.objectiveNames = objectiveNames;
        this.constraintNames = constraintNames;
        this.isCDTLZ = isCDTLZ;
        this.coevolutionaryRunNumber = coevolutionaryRunNumber;
        this.weightOfWeights = weightOfWeights;
        this.moeaChoice = moeaChoice;
        this.problemChoice = problemChoice;
        this.initialPopulation = new Population();
        this.finalPopulation = new Population();
    }

    public Population runMOEA() throws Exception {
        Result result;
        if (weightOfWeights) {
            result = new Result(saveDirectory, constraintNames.length+1);
        } else {
            result = new Result(saveDirectory, constraintNames.length);
        }

        // Run MOEA
        long startTime = System.currentTimeMillis();
        algorithm.step();

        ArrayList<Solution> allSolutions = new ArrayList<>();
        if (this.moeaChoice.equals("MOEAD_")) {
            this.initialPopulation = new Population(this.initialization.initialize());
        } else {
            this.initialPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
        }
        for (Solution currentSolution : this.initialPopulation) {
            currentSolution.setAttribute("NFE",0);
            allSolutions.add(currentSolution);
        }

        while (!algorithm.isTerminated() && (algorithm.getNumberOfEvaluations() < internalMaxEvaluations)) {
            try {
                algorithm.step();
            } catch (Exception e) {
                e.printStackTrace();
            }

            if (moeaChoice.equals("EpsilonMOEA_")) { // Epsilon-MOEA
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
        System.out.println("Internal Population Evolution Done in " + ((endTime - startTime)/1000.0) + " s");

        // Save solutions to CSV file
        if (!this.moeaChoice.equals("MOEAD_")) {
            this.finalPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
            this.finalArchive =  ((AbstractEvolutionaryAlgorithm) algorithm).getArchive();
        } else {
            this.finalArchive =  algorithm.getResult();
        }

        StringJoiner weightsSJ = new StringJoiner("_");
        for (int i = 0; i < constraintNames.length; i++) {
            weightsSJ.add("w" + i + "-" + Double.toString(Precision.round(constraintWeights[i], 5)).replace(".",";"));
        }
        if (weightOfWeights) {
            weightsSJ.add("ww-" + Double.toString(Precision.round(constraintWeights[constraintNames.length], 5)).replace(".",";"));
        }
        String populationFilename = "run " + coevolutionaryRunNumber + File.separator + problemChoice + moeaChoice + "Constr_Weights-" + weightsSJ.toString() + "_Coev_nfe-" + currentCoevolutionNFE + "_" + "finalpop" + ".csv";
        String archiveFilename = "run " + coevolutionaryRunNumber + File.separator + problemChoice + moeaChoice+ "Constr_Weights-" + weightsSJ.toString() + "_Coev_nfe-" + currentCoevolutionNFE + "_" + "finalarchive" + ".csv";
        String allSolutionsFilename = "run " + coevolutionaryRunNumber + File.separator + problemChoice + moeaChoice + "Constr_Weights-" + weightsSJ.toString() + "_Coev_nfe-" + currentCoevolutionNFE + "_allSolutions.csv";

        result.saveAllInternalSolutions(allSolutionsFilename, allSolutions, variableNames, objectiveNames, constraintNames, isCDTLZ, false);
        result.saveInternalPopulationOrArchive(populationFilename, this.finalPopulation, variableNames, objectiveNames, constraintNames, isCDTLZ, false);
        result.saveInternalPopulationOrArchive(archiveFilename, this.finalArchive, variableNames, objectiveNames, constraintNames, isCDTLZ, false);

        return getFinalPopulation();
    }

    private Population getFinalPopulation() {
        // Call after internal population evolution is complete!
        if (this.moeaChoice.equals("MOEAD_")) { // MOEA-D
            // Replace last members of initial population with archive
            this.finalPopulation = new Population();
            for (int i = 0; i < (this.initialPopulation.size() - this.finalArchive.size()); i++) {
                this.finalPopulation.add(this.initialPopulation.get(i));
            }
            this.finalPopulation.addAll(this.finalArchive);

            return this.finalPopulation;
        } else {
            return this.finalPopulation;
        }
    }


}
