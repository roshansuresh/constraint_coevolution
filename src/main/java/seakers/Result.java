package seakers;

import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.EncodingUtils;
import seakers.datastructure.ConstraintWeightsData;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringJoiner;

public class Result {

    private final String saveDirectory;
    private final int numberOfWeights;

    public Result(String saveDirectory, int numberOfWeights) {
        this.saveDirectory = saveDirectory;
        this.numberOfWeights = numberOfWeights;
    }

    public void fillConstraintWeightsData(ConstraintWeightsData weightsData, Solution solution, int nfe, int numberOfWeightsObjectives, int numberOfConstraints) {
        double[] solutionWeights = new double[numberOfWeights];
        for (int i = 0; i < solution.getNumberOfVariables()-1; i++) {
            solutionWeights[i] = EncodingUtils.getReal(solution.getVariable(i));
        }

        weightsData.setConstraintWeights(solutionWeights);
        weightsData.setNFE(nfe);
        double[] solutionFitness = new double[numberOfWeightsObjectives];
        for (int i = 0; i < numberOfWeightsObjectives; i++) {
            solutionFitness[i] = solution.getObjective(i);
        }
        weightsData.setFitness(solutionFitness);
        if (numberOfConstraints > 0) {
            weightsData.setNumberOfFeasibleSolutions((int) solution.getAttribute("Number of Feasible Solutions"));
        }
    }

    public void saveConstraintWeights(ArrayList<ConstraintWeightsData> weightsSet, int numberOfWeightsObjectives, int numberOfConstraints, String filename, boolean saveNFE) {
        String fullFilename = saveDirectory + File.separator + filename;
        File saveFile = new File(fullFilename);
        saveFile.getParentFile().mkdirs();

        System.out.println("Saving Evaluated Constraint Weights");

        try (FileWriter writer = new FileWriter(saveFile)) {
            StringJoiner headings = new StringJoiner(",");
            if (saveNFE) {
                headings.add("NFE");
            }
            for (int i = 0; i < numberOfWeights; i++) {
                headings.add("Weight: Constraint " + i);
            }
            for (int i = 0; i < numberOfWeightsObjectives; i++) {
                headings.add("Fitness Value " + i);
            }
            if (numberOfConstraints > 0) {
                headings.add("Number of Feasible Solutions");
            }

            writer.append(headings.toString());
            writer.append("\n");

            for (ConstraintWeightsData weightsData : weightsSet) {
                StringJoiner weightsSJ = new StringJoiner(",");
                if (saveNFE) {
                    weightsSJ.add(Integer.toString(weightsData.getNFE()));
                }
                double[] weights = weightsData.getConstraintWeights();
                for (int i = 0; i < numberOfWeights; i++) {
                    weightsSJ.add(Double.toString(weights[i]));
                }
                double[] solutionFitness = weightsData.getFitness();
                for (int i = 0; i < numberOfWeightsObjectives; i++) {
                    weightsSJ.add(Double.toString(solutionFitness[i]));
                }
                if (numberOfConstraints > 0) {
                    weightsSJ.add(Integer.toString(weightsData.getNumberOfFeasibleSolutions()));
                }
                writer.append(weightsSJ.toString());
                writer.append("\n");
            }
            writer.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveFinalPopulationOrArchive(Population population, String filename, int numberOfWeightsObjectives, int numberOfConstraints) {
        String fullFilename = saveDirectory + File.separator + filename;
        File saveFile = new File(fullFilename);
        saveFile.getParentFile().mkdirs();

        System.out.println("Saving Final Population and Archive");

        try (FileWriter writer = new FileWriter(saveFile)) {
            StringJoiner headings = new StringJoiner(",");
            for (int i = 0; i < numberOfWeights; i++) {
                headings.add("Weight: Constraint " + i);
            }
            for (int i = 0; i < numberOfWeightsObjectives; i++) {
                headings.add("Fitness Value " + i);
            }
            if (numberOfConstraints > 0) {
                headings.add("Number of Feasible Solutions");
            }

            writer.append(headings.toString());
            writer.append("\n");

            for (Solution solution : population) {
                StringJoiner weightsSJ = new StringJoiner(",");
                for (int i = 0; i < numberOfWeights; i++) {
                    weightsSJ.add(Double.toString(EncodingUtils.getReal(solution.getVariable(i))));
                }
                for (int i = 0; i < numberOfWeightsObjectives; i++) {
                    weightsSJ.add(Double.toString(solution.getObjective(i)));
                }
                if (numberOfConstraints > 0) {
                    weightsSJ.add(Integer.toString((int) solution.getAttribute("Number of Feasible Solutions")));
                }
                writer.append(weightsSJ.toString());
                writer.append("\n");
            }
            writer.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveAllInternalSolutions(String filename, ArrayList<Solution> solutionSet, String[] variableNames, String[] objectiveNames, String[] constraintNames, boolean isCDTLZ, boolean isMOEA) {

        String fullFilename = saveDirectory + File.separator + filename;
        File saveFile = new File(fullFilename);
        saveFile.getParentFile().mkdirs();

        System.out.println("Saving solutions");

        try(FileWriter writer = new FileWriter(saveFile)) {
            StringJoiner headings = new StringJoiner(",");
            headings.add("NFE");
            for (String variableName : variableNames) {
                headings.add(variableName);
            }
            for (String objectiveName : objectiveNames) {
                headings.add(objectiveName);
            }
            for (String constraintName : constraintNames) {
                headings.add(constraintName);
            }
            writer.append(headings.toString());
            writer.append("\n");

            for (Solution solution : solutionSet) {
                StringJoiner sj = new StringJoiner(",");
                sj.add(Integer.toString((Integer) solution.getAttribute("NFE")));
                if (isCDTLZ) { // Only for the CDTLZ problems, extract real variables and store
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        sj.add(Double.toString(EncodingUtils.getReal(solution.getVariable(i))));
                    }
                } else { // Metamaterial problems use binary variables
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        sj.add(Boolean.toString(EncodingUtils.getBoolean(solution.getVariable(i))));
                    }
                }
                for (int i = 0; i < solution.getNumberOfObjectives(); i++) { // OBJECTIVE NAMES ATTRIBUTE NOT STORED IN C_DTLZ MOEA RUNS, SWITCH TO DIRECTLY OBTAINING OBJECTIVES FROM EACH SOLUTION FOR THAT CASE
                    sj.add(Double.toString((Double) solution.getAttribute(objectiveNames[i])));
                }
                for (int i = 0; i < solution.getNumberOfConstraints(); i++) {
                    sj.add(Double.toString(solution.getConstraint(i)));
                }
                writer.append(sj.toString());
                writer.append("\n");
            }
            writer.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveInternalPopulationOrArchive (String filename, Population population, String[] variableNames, String[] objectiveNames, String[] constraintNames, boolean isCDTLZ, boolean isMOEA) {

        String fullFilename = saveDirectory + File.separator + filename;
        File saveFile = new File(fullFilename);
        saveFile.getParentFile().mkdirs();

        System.out.println("Saving population or archive");

        try(FileWriter writer = new FileWriter(saveFile)) {
            StringJoiner headings = new StringJoiner(",");
            for (String variableName : variableNames) {
                headings.add(variableName);
            }
            for (String objectiveName : objectiveNames) {
                headings.add(objectiveName);
            }
            for (String constraintName : constraintNames) {
                headings.add(constraintName);
            }
            writer.append(headings.toString());
            writer.append("\n");

            for (Solution solution : population) {
                StringJoiner sj = new StringJoiner(",");
                if (isCDTLZ) { // Only for the CDTLZ problems, extract real variables and store
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        sj.add(Double.toString(EncodingUtils.getReal(solution.getVariable(i))));
                    }
                } else { // Metamaterial problems use binary variables
                    for (int i = 0; i < solution.getNumberOfVariables(); i++) {
                        sj.add(Boolean.toString(EncodingUtils.getBoolean(solution.getVariable(i))));
                    }
                }

                for (int i = 0; i < solution.getNumberOfObjectives(); i++) { // OBJECTIVE NAMES ATTRIBUTE NOT STORED IN C_DTLZ MOEA RUNS, SWITCH TO DIRECTLY OBTAINING OBJECTIVES FROM EACH SOLUTION FOR THAT CASE
                    sj.add(Double.toString((Double) solution.getAttribute(objectiveNames[i])));
                }
                for (int i = 0; i < solution.getNumberOfConstraints(); i++) {
                    sj.add(Double.toString(solution.getConstraint(i)));
                }
                writer.append(sj.toString());
                writer.append("\n");
            }
            writer.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
