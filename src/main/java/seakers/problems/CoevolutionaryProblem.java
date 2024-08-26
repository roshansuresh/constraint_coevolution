package seakers.problems;

import org.moeaframework.algorithm.AbstractEvolutionaryAlgorithm;
import org.moeaframework.algorithm.EpsilonMOEA;
import org.moeaframework.algorithm.IBEA;
import org.moeaframework.algorithm.MOEAD;
import org.moeaframework.core.*;
import org.moeaframework.core.comparator.DominanceComparator;
import org.moeaframework.core.comparator.ParetoDominanceComparator;
import org.moeaframework.core.comparator.ParetoObjectiveComparator;
import org.moeaframework.core.indicator.Hypervolume;
import org.moeaframework.core.operator.InjectedInitialization;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.AbstractProblem;
import seakers.Result;
import seakers.datastructure.ConstraintWeightsData;
import seakers.utils.ConstrainedIndicatorFitnessEvaluator;
import seakers.utils.HypervolumeDifferenceFitnessEvaluator;
import seakers.utils.ObjectiveHypervolume;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class CoevolutionaryProblem extends AbstractProblem {

    private AbstractProblem optimizationProblem;
    private String saveDirectory;
    private int internalMaxEvaluations;
    private int currentCoevolutionNFE;
    private Population internalPopulation;
    private EpsilonBoxDominanceArchive internalArchive;
    private final DominanceComparator internalComparator;
    private final Selection internalSelection;
    private final Variation internalVariation;
    private final Hypervolume internalHypervolumeComputer; // Used to compute fitness
    private final boolean CMOEA; // Whether the coevolutionary optimization is single or multi objective
    private final boolean evolvePopulation;
    private final boolean integerWeights;
    private final String[] variableNames;
    private final String[] objectiveNames;
    private final String[] constraintNames;
    private ArrayList<ConstraintWeightsData> allWeights;
    private final int coevolutionaryRunNumber;
    private final int numberOfCoevolutionaryObjectives;
    private Result result;
    private final boolean isCDTLZ;
    private final int coevolutionaryPopulationSize;
    private Population bestInternalPopulation;
    private double bestInitialIndicator;
    private final boolean weightOfWeights;
    private final int moeaChoice;
    private final String moeaName;
    private final String problemChoice;

    public CoevolutionaryProblem(AbstractProblem optimizationProblem, String saveDirectory, int coevolutionaryPopulationSize, int internalMaxEvaluations, Population internalPopulation, EpsilonBoxDominanceArchive internalArchive, DominanceComparator internalComparator, Selection internalSelection, Variation internalVariation, Hypervolume internalHypervolumeComputer, boolean CMOEA, boolean evolvePopulation, boolean integerWeights, boolean weightOfWeights, String[] variableNames, String[] objectiveNames, String[] constraintNames, int numberOfCoevolutionaryObjectives, Result result, boolean isCDTLZ, int moeaChoice, int coevolutionaryRunNumber, String moeaName, String problemChoice) {
        super(constraintNames.length, numberOfCoevolutionaryObjectives);
        this.optimizationProblem = optimizationProblem;
        this.saveDirectory = saveDirectory;
        this.coevolutionaryPopulationSize = coevolutionaryPopulationSize;
        this.internalMaxEvaluations = internalMaxEvaluations;
        this.internalPopulation = internalPopulation;
        this.internalArchive = internalArchive;
        this.internalComparator = internalComparator;
        this.internalSelection = internalSelection;
        this.internalVariation = internalVariation;
        this.internalHypervolumeComputer = internalHypervolumeComputer;
        this.CMOEA = CMOEA;
        this.evolvePopulation = evolvePopulation;
        this.integerWeights = integerWeights;
        this.variableNames = variableNames;
        this.objectiveNames = objectiveNames;
        this.constraintNames = constraintNames;
        this.isCDTLZ = isCDTLZ;
        this.moeaChoice = moeaChoice;
        this.coevolutionaryRunNumber = coevolutionaryRunNumber;
        this.result = result;
        this.numberOfCoevolutionaryObjectives = numberOfCoevolutionaryObjectives;
        this.allWeights = new ArrayList<>();
        this.currentCoevolutionNFE = 0;
        this.bestInitialIndicator = 0.0;
        this.bestInternalPopulation = new Population();
        this.weightOfWeights = weightOfWeights;
        this.moeaName = moeaName;
        this.problemChoice = problemChoice;
        //pool = Executors.newFixedThreadPool(numberOfCores);
        //cs = new ExecutorCompletionService<>(pool);
    }

    @Override
    public void evaluate(Solution solution) {
        // Modify constraint weights in the problem class
        double weightMultiplier = 1.0;
        int startLocation = 0;
        int endLocation = constraintNames.length;
        if (weightOfWeights) {
            weightMultiplier = EncodingUtils.getReal(solution.getVariable(0));
            startLocation = 1;
            endLocation = constraintNames.length+1;
        }

        boolean allLowestWeights = false;

        double[] constraintWeights = new double[endLocation];
        int index = 0;
        for (int i = startLocation; i < endLocation; i++) {
            if (integerWeights) {
                constraintWeights[index] = ((int) EncodingUtils.getReal(solution.getVariable(i)))*weightMultiplier;
            } else {
                if (allLowestWeights) {
                    constraintWeights[index] = 0;
                } else {
                    constraintWeights[index] = (Math.pow(10, EncodingUtils.getReal(solution.getVariable(i))))*weightMultiplier; // weight = 10^x (*weightOfWeights if applicable)
                }
            }
            index++;
        }
        if (weightOfWeights) {
            constraintWeights[endLocation-1] = weightMultiplier;
        }

        ((AbstractInternalProblem) optimizationProblem).setConstraintWeights(constraintWeights);

        // Initialize and run internal MOEA to propagate the initial population
        List<Solution> injectionSolutions = new ArrayList<>();
        for (Solution value : internalPopulation) {
            injectionSolutions.add(value);
        }
        Initialization initialization = new InjectedInitialization(optimizationProblem, internalPopulation.size(), injectionSolutions);

        Population population = new Population();
        Algorithm moea = null;
        switch (moeaChoice) {
            case 1:
                moea = new EpsilonMOEA(optimizationProblem, population, internalArchive, internalSelection, internalVariation, initialization, internalComparator);
                break;
            case 2:
                int neighbourhoodSize = 5; // Number of neighbourhood problems used for mating
                double delta = 1; // probability of using only neighbourhood problems for mating instead of entire population
                double eta = 1; // maximum number of population slots a new solution will replace
                int updateUtility = 5; // frequency (generations) in which utility values are updated
                moea = new MOEAD(optimizationProblem, neighbourhoodSize, initialization, internalVariation, delta, eta, updateUtility);
                break;
            case 3:
                //ConstrainedIndicatorFitnessEvaluator ife = new ConstrainedIndicatorFitnessEvaluator(optimizationProblem, new ParetoDominanceComparator());

                double[] normalizedObjectivesMaximum = new double[optimizationProblem.getNumberOfObjectives()];
                double[] normalizedObjectivesMinimum = new double[optimizationProblem.getNumberOfObjectives()];
                Arrays.fill(normalizedObjectivesMaximum, 1.0);
                Arrays.fill(normalizedObjectivesMinimum, 0.0);

                HypervolumeDifferenceFitnessEvaluator ife = new HypervolumeDifferenceFitnessEvaluator(optimizationProblem, normalizedObjectivesMinimum, normalizedObjectivesMaximum);


                moea = new IBEA(optimizationProblem, internalArchive, initialization, internalVariation, ife);
                break;
            default:
                System.out.println("Invalid MOEA choice");
                break;
        }

        // Run MOEA
        System.out.println("Starting Internal Population Evolution");
        //cs.submit();
        ModifiedProblemSearch internalMOEA = new ModifiedProblemSearch(moea, initialization, saveDirectory, constraintWeights, internalMaxEvaluations, currentCoevolutionNFE, variableNames, objectiveNames, constraintNames, isCDTLZ, weightOfWeights, coevolutionaryRunNumber, moeaName, problemChoice);
        Population finalPopulation = new Population();

        try {
            finalPopulation = internalMOEA.runMOEA();
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Compute fitness value based on final population
        boolean feasibleHypervolumeDifference = false; // Only matters if cMOEA is false

        ArrayList<Solution> feasibleSolutions = getFeasibleSolutions(finalPopulation);
        if (optimizationProblem.getNumberOfConstraints() > 0) {
            solution.setAttribute("Number of Feasible Solutions", feasibleSolutions.size());
        }

        double initialHypervolume = 0.0;
        double hypervolume = 0.0;
        ArrayList<Solution> initialFeasibleSolutions = getFeasibleSolutions(internalPopulation);

        if (CMOEA || feasibleHypervolumeDifference) {

            // Compute initial hypervolume
            if (initialFeasibleSolutions.size() > 0) {
                NondominatedPopulation initialNondominatedPopulation = new NondominatedPopulation(internalComparator);
                for (Solution internalSolution : initialFeasibleSolutions) {
                    initialNondominatedPopulation.add(internalSolution);
                }
                initialHypervolume = internalHypervolumeComputer.evaluate(initialNondominatedPopulation);
            }

            if (feasibleSolutions.size() > 0) {
                NondominatedPopulation nondominatedPopulation = new NondominatedPopulation(internalComparator); // Comparing only in terms of Pareto Objective Dominance
                for (Solution feasibleSolution : feasibleSolutions) {
                    nondominatedPopulation.add(feasibleSolution);
                }
                hypervolume = internalHypervolumeComputer.evaluate(nondominatedPopulation);
            }
        }

        double populationUpdateIndicator = 0.0;
        double averageFitness = 0.0;

        if (CMOEA) {
            double[] averageObjectives = new double[optimizationProblem.getNumberOfObjectives()];
            for (Solution internalSolution : feasibleSolutions) {
                for (int i = 0; i < optimizationProblem.getNumberOfObjectives(); i++) {
                    averageObjectives[i] -= internalSolution.getObjective(i); // all DTLZ objectives must be minimized, so negative of sum of objectives of all feasible solutions must be minimized
                }
            }

            for (int i = 0; i < optimizationProblem.getNumberOfObjectives(); i++) {
                averageObjectives[i] /= (feasibleSolutions.size() + 1e-5);
                averageObjectives[i] -= feasibleSolutions.size();

                solution.setObjective(i, averageObjectives[i]);
            }

            populationUpdateIndicator = hypervolume - initialHypervolume;

        } else if (feasibleHypervolumeDifference) {
            averageFitness = -(hypervolume - initialHypervolume); // Previous hypervolume is subtracted to isolate HV improvement because of the current constraint weights

            // Include next two lines for Coello-like fitness
            //averageFitness /= (feasibleSolutions.size() + 1e-5); // To make sure not to divide by zero
            //averageFitness -= feasibleSolutions.size(); // average fitness is to be maximized but a minimizing comparator is used in the Coevolutionary MOEA (due to FitnessIndicator being deprecated)

            solution.setObjective(0, averageFitness);

            populationUpdateIndicator = hypervolume - initialHypervolume;

        } else {
            Population nonNaNFinalPopulation = new Population();
            for (Solution sol : finalPopulation) {
                boolean naNObjective = false;
                for (int i = 0; i < objectiveNames.length; i++) {
                    double objective = (double) sol.getAttribute(objectiveNames[i]);
                    if (Double.isNaN(objective)) {
                        naNObjective = true;
                        break;
                    }
                }
                if (!naNObjective) {
                    // Create copy of solution with objectiveNames as solution objectives and add to nonNaNFinalPopulation
                    Solution trueObjectiveSol = sol.deepCopy();
                    for (int i = 0; i < objectiveNames.length; i++) {
                        trueObjectiveSol.setObjective(i, (Double) sol.getAttribute(objectiveNames[i]));
                    }
                    nonNaNFinalPopulation.add(sol);
                }
            }

            NondominatedPopulation nondominatedPopulation = new NondominatedPopulation(new ParetoObjectiveComparator()); // Comparing only in terms of Pareto Objective Dominance
            nondominatedPopulation.addAll(nonNaNFinalPopulation);

            // Assuming problem objectives are normalized (otherwise use objective minimum and maximum from executable class as parameters to this class)
            double[] minimum = new double[optimizationProblem.getNumberOfObjectives()]; //= new double[]{-1, 0}; // Both metamaterial problems involve maximizing the first objective and minimizing the second
            double[] maximum = new double[optimizationProblem.getNumberOfObjectives()];; //= new double[]{0, 1};
            if (isCDTLZ) { // All C-DTLZ problems have objectives to minimize
                Arrays.fill(minimum, 0);
                Arrays.fill(maximum, 1);
            } else {
                minimum = new double[]{-1, 0}; // Both metamaterial problems involve maximizing the first objective and minimizing the second
                maximum = new double[]{0, 1};
            }
            //double[] minimum = new double[optimizationProblem.getNumberOfObjectives()];
            //double[] maximum = new double[optimizationProblem.getNumberOfObjectives()];
            //Arrays.fill(minimum, -1.0);
            //Arrays.fill(maximum, 1.0);

            //Hypervolume objectiveHypervolumeComputer = new Hypervolume(optimizationProblem, minimum, maximum); // Hypervolume's evaluate method does not consider constraint violatinmg solutions, so custom ObjectiveHypervolume class instance is used
            ObjectiveHypervolume objectiveHypervolumeComputer = new ObjectiveHypervolume(optimizationProblem, minimum, maximum);
            double objectiveHypervolume = objectiveHypervolumeComputer.computeObjectiveHypervolume(nondominatedPopulation);

            NondominatedPopulation initialParetoFront = new NondominatedPopulation(new ParetoObjectiveComparator());
            for (Solution sol : internalPopulation) {
                boolean naNObjective = false;
                for (int i = 0; i < objectiveNames.length; i++) {
                    double objective = (double) sol.getAttribute(objectiveNames[i]);
                    if (Double.isNaN(objective)) {
                        naNObjective = true;
                        break;
                    }
                }
                if (!naNObjective) {
                    // Create copy of solution with objectiveNames as solution objectives and add to initialParetoFront
                    Solution trueObjectiveSol = sol.deepCopy();
                    for (int i = 0; i < objectiveNames.length; i++) {
                        trueObjectiveSol.setObjective(i, (Double) sol.getAttribute(objectiveNames[i]));
                    }
                    initialParetoFront.add(sol);
                }
            }
            double initialObjectiveHypervolume = objectiveHypervolumeComputer.computeObjectiveHypervolume(initialParetoFront);

            double constraintViolationWeight = 10*Math.exp(-feasibleSolutions.size());
            //double initialFitness = (getAggregateConstraintViolation(initialParetoFront))/(initialFeasibleSolutions.size() + 1e-5) - initialObjectiveHypervolume;

            averageFitness = constraintViolationWeight*(getAggregateConstraintViolation(finalPopulation) - getAggregateConstraintViolation(internalPopulation)) - (objectiveHypervolume - initialObjectiveHypervolume);
            solution.setObjective(0, averageFitness);
            populationUpdateIndicator = -averageFitness;
        }

        // Update best initial internal indicator if necessary (since populationUpdateIndicator = -averageFitness, larger value is better)
        boolean bestInitialFitnessUpdated = false;
        if (populationUpdateIndicator > bestInitialIndicator) {
            bestInitialIndicator = populationUpdateIndicator;
            bestInitialFitnessUpdated = true;
        }

        // Check if the constraint weights are part of the initial population
        boolean initialPopulationWeights = false;
        if (currentCoevolutionNFE < coevolutionaryPopulationSize) {
            initialPopulationWeights = true;
        }

        // Update best internal population corresponding to the initial population of weights
        if ((bestInitialFitnessUpdated) && (initialPopulationWeights)) {
            bestInternalPopulation = finalPopulation;
        }

        // After initial population of weights is evaluated, update the internal population with the best internal population for subsequent weights
        if ((currentCoevolutionNFE == (coevolutionaryPopulationSize-1)) && (evolvePopulation)) { // note that the weights at (coevolutionaryPopulationSize-1) are already evaluated so this update is for after the initial population of weights
            if (bestInternalPopulation.size() > 0) { // if bestInternalPopulation is empty, that implies a better population could not be obtained during the evaluation of the initial weights population
                setInternalPopulation(bestInternalPopulation);
            }
        }

        // Internal population is updated based on ``evolvePopulation'' and fitness values for subsequent weights
        if (EncodingUtils.getBoolean(solution.getVariable(solution.getNumberOfVariables()-1)) && (populationUpdateIndicator > 0.0)) {
            setInternalPopulation(finalPopulation); // The initial population for the evaluation of the next set of weights is the final population of the previous set of weights
        }
        BinaryVariable var = new BinaryVariable(1);
        if (evolvePopulation) {
            EncodingUtils.setBoolean(var, true); // subsequent solutions will use the updated internal population
        } else {
            EncodingUtils.setBoolean(var, false);
        }

        solution.setVariable(endLocation, var);

        // Save current constraint weights to arraylist
        ConstraintWeightsData constraintWeightsData;
        if (weightOfWeights) {
            constraintWeightsData = new ConstraintWeightsData(constraintNames.length+1, numberOfCoevolutionaryObjectives);
        } else {
            constraintWeightsData = new ConstraintWeightsData(constraintNames.length, numberOfCoevolutionaryObjectives);
        }
        result.fillConstraintWeightsData(constraintWeightsData, solution, currentCoevolutionNFE, numberOfCoevolutionaryObjectives, optimizationProblem.getNumberOfConstraints());
        allWeights.add(constraintWeightsData);
        currentCoevolutionNFE++;

        //cs = null;
        //pool.shutdown();

    }

    @Override
    public Solution newSolution() {
        int numberOfObjectives = 1;
        if (CMOEA) {
            numberOfObjectives = optimizationProblem.getNumberOfObjectives();
        }
        Solution newRandomSolution;
        int startLocation = 0;
        int endLocation = constraintNames.length;
        if (weightOfWeights) {
            newRandomSolution = new Solution(constraintNames.length+2, numberOfObjectives);
            startLocation = 1;
            endLocation = constraintNames.length+1;
        } else {
            newRandomSolution = new Solution(constraintNames.length+1, numberOfObjectives);
        }
        //double[] newRandomWeights = new double[numberOfConstraints];

        if (weightOfWeights) {
            double randomWeightOfWeights = PRNG.nextDouble();
            RealVariable weightVar = new RealVariable(randomWeightOfWeights, 0.0, 1.0);
            newRandomSolution.setVariable(0, weightVar);
        }

        RealVariable var;
        for (int i = startLocation; i < endLocation; i++) {
            if (integerWeights) {
                int newRandomWeight = PRNG.nextInt(11);
                //newRandomWeights[i] = newRandomWeight;
                var = new RealVariable(newRandomWeight, 0, 10);
                newRandomSolution.setVariable(i, var);
            } else {
                double newRandomWeight = PRNG.nextDouble();
                //newRandomWeights[i] = newRandomWeight;
                var = new RealVariable(newRandomWeight, -3.0, 1.0);
                newRandomSolution.setVariable(i, var);
            }
        }

        // Binary decision signifying whether the updated population must be used
        BinaryVariable binaryVariable = new BinaryVariable(1);
        EncodingUtils.setBoolean(binaryVariable, false); // initial constraint weights will not use the updated population but subsequent weights will
        newRandomSolution.setVariable(endLocation, binaryVariable);

        return newRandomSolution;
    }

    public AbstractProblem getOptimizationProblem() {
        return this.optimizationProblem;
    }

    private Population getInternalPopulation() {
        return this.internalPopulation;
    }

    private void setInternalPopulation(Population population) {
        this.internalPopulation = population;
    }

    public ArrayList<ConstraintWeightsData> getAllWeights() { return this.allWeights; }

    public boolean getCMOEA() {return this.CMOEA; }

    private ArrayList<Solution> getFeasibleSolutions(Population population) {
        ArrayList<Solution> feasibleSolutions = new ArrayList<>();
        Iterator<Solution> iter = population.iterator();
        while (iter.hasNext()) {
            Solution currentSolution = iter.next();
            if (!currentSolution.violatesConstraints()) { // According to any Cx_DTLZy class, constraint = 0 if actual value (c) is >= 0, else it is c
                for (double objective : currentSolution.getObjectives()) { // Don't include solution if any objective is NaN (happens in metamaterial problems)
                    if (Double.isNaN(objective)) {
                        continue;
                    }
                }

                // Copying all attributes and parameters to a new Solution object, except the objectives are the true optimization objectives without constraint penalization
                // These objectives are used to compute the fitness for the corresponding constraint weights
                Solution trueObjectiveSolution = currentSolution.copy();
                boolean invalidTrueObjectives = false;
                for (int i = 0; i < currentSolution.getNumberOfObjectives(); i++) {
                    double currentTrueObjective = (double) currentSolution.getAttribute(objectiveNames[i]);
                    if (Double.isNaN(currentTrueObjective)) { // Possible for the metamaterial problems where designs with invalid stiffness values have bad optimization objectives but True Objective 1 is set to NaN
                        invalidTrueObjectives = true;
                        break;
                    } else {
                        trueObjectiveSolution.setObjective(i, currentTrueObjective);
                    }
                }
                if (!invalidTrueObjectives) {
                    feasibleSolutions.add(trueObjectiveSolution);
                }
            }
        }
        return feasibleSolutions;
    }

    private double getAggregateConstraintViolation(Iterable<Solution> solutions) {
        double aggregateConstraintViolation = 0.0;
        int solutionsCounter = 0;
        for (Solution currentSolution : solutions) {
            double aggregateSolutionConstraintViolation = 0.0;
            for (int k = 0; k < currentSolution.getNumberOfConstraints(); k++) {
                aggregateSolutionConstraintViolation += currentSolution.getConstraint(k);
            }
            aggregateConstraintViolation += aggregateSolutionConstraintViolation/currentSolution.getNumberOfConstraints();
            solutionsCounter++;
        }
        aggregateConstraintViolation /= solutionsCounter;

        return aggregateConstraintViolation;
    }
}
