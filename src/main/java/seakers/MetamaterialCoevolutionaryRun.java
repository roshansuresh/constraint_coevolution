package seakers;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.algorithm.EpsilonMOEA;
import org.moeaframework.algorithm.IBEA;
import org.moeaframework.algorithm.MOEAD;
import org.moeaframework.algorithm.single.AggregateObjectiveComparator;
import org.moeaframework.algorithm.single.GeneticAlgorithm;
import org.moeaframework.algorithm.single.LinearObjectiveComparator;
import org.moeaframework.core.*;
import org.moeaframework.core.comparator.DominanceComparator;
import org.moeaframework.core.comparator.ParetoDominanceComparator;
import org.moeaframework.core.comparator.ParetoObjectiveComparator;
import org.moeaframework.core.indicator.Hypervolume;
import org.moeaframework.core.operator.CompoundVariation;
import org.moeaframework.core.operator.InjectedInitialization;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.operator.real.PM;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.AbstractProblem;
import seakers.operators.SimulatedBinaryWeightsCrossover;
import seakers.operators.UniformIntegerWeightsMutation;
import seakers.operators.UniformWeightsCrossover;
import seakers.problems.CoevolutionaryProblem;
import seakers.problems.metamaterial.ModifiedArteryProblem;
import seakers.problems.metamaterial.ModifiedEqualStiffnessProblem;
import seakers.trussaos.initialization.SynchronizedMersenneTwister;
import seakers.utils.ConstrainedIndicatorFitnessEvaluator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

public class MetamaterialCoevolutionaryRun {

    public static ExecutorService pool;
    public static CompletionService<Algorithm> ecs;
    /**
     * Matlab Engine for function evaluation
     */
    private static MatlabEngine engine;

    public static void main (String[] args) throws InterruptedException, ExecutionException {

        // Save location
        //String saveDir = System.getProperty("user.dir") + File.separator + "results"; // File not found error!
        String saveDir = "C:\\SEAK Lab\\SEAK Lab Github\\Coevolution-based Constraint Satisfaction\\results";

        engine = MatlabEngine.startMatlab();

        String matlabScriptsLocation = "C:\\SEAK Lab\\SEAK Lab Github\\Coevolution-based Constraint Satisfaction\\matlab";
        engine.feval("addpath", matlabScriptsLocation); // Add location of MATLAB scripts used to compute objectives, constraints and heuristics to MATLAB's search path

        PRNG.setRandom(new SynchronizedMersenneTwister());

        int numCPU = 1;
        int numRuns = 1;
        pool = Executors.newFixedThreadPool(numCPU);
        ecs = new ExecutorCompletionService<>(pool);

        boolean cMOEA = false; // if the coevolutionary optimization is single or multi objective
        boolean evolveInternalPopulation = true; // if initial internal population for subsequent heuristic weights should be updated or not
        boolean integerWeights = false; // if formulation for the heuristic weights should be integer or real
        boolean weightOfWeights = false; // if an additional weight multiplicative parameter design decision for the outer GA is used
        boolean periodicZeroInjection = false; // if zero solution is added for evaluation to population at different NFE

        int moeaChoice = 1; // choice of moea algorithm: 1 -> epsilon-MOEA, 2 -> MOEA-D. 2 -> IBEA

        // Problem
        //int numberOfHeuristics = 4; // 4 heuristics for the metamaterial problems

        /**
         * modelChoice = 0 --> Fibre Stiffness Model
         *             = 1 --> Truss Stiffness Model
         *             = 2 --> Beam Model
         */
        int modelChoice = 1; // Fibre stiffness model cannot be used for the artery problem

        boolean arteryProblem = false; // Solve the artery optimization (otherwise the original truss problem is solved)
        double targetStiffnessRatio = 1;
        if (arteryProblem) {
            targetStiffnessRatio = 0.421;
        }

        // Heuristic Enforcement Methods
        /**
         * partialCollapsibilityConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         * nodalPropertiesConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         * orientationConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         * intersectionConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
         *
         * heuristicsConstrained = [partialCollapsibilityConstrained, nodalPropertiesConstrained, orientationConstrained, intersectionConstrained]
         */
        boolean[] partialCollapsibilityConstrained = {false, false, false, false, false, false, false};
        boolean[] nodalPropertiesConstrained = {false, false, false, false, false, false, false};
        boolean[] orientationConstrained = {false, false, false, false, false, false, false};
        boolean[] intersectionConstrained = {false, false, false, false, false, false, false};

        boolean[][] heuristicsConstrained = new boolean[4][7];
        for (int i = 0; i < 7; i++) {
            heuristicsConstrained[0][i] = partialCollapsibilityConstrained[i];
            heuristicsConstrained[1][i] = nodalPropertiesConstrained[i];
            heuristicsConstrained[2][i] = orientationConstrained[i];
            heuristicsConstrained[3][i] = intersectionConstrained[i];
        }

        // Dimensions for printable solutions
        double printableRadius = 250e-6; // in m
        double printableSideLength = 10e-3; // in m
        double printableModulus = 1.8162e6; // in Pa
        double sideNodeNumber = 3.0D;
        int nucFactor = 3; // Not used if PBC model is used

        String[] constraintNames;
        String problemChoice;
        if (arteryProblem) {
            constraintNames = new String[]{"FeasibilityViolation", "ConnectivityViolation"};
            problemChoice = "artery_";
        } else {
            constraintNames = new String[]{"FeasibilityViolation", "ConnectivityViolation", "StiffnessRatioViolation"};
            problemChoice = "equalstiffness_";
        }

        int totalNumberOfMembers;
        if (sideNodeNumber >= 5) {
            int sidenumSquared = (int) (sideNodeNumber*sideNodeNumber);
            totalNumberOfMembers =  sidenumSquared * (sidenumSquared - 1)/2;
        }
        else {
            totalNumberOfMembers = (int) (CombinatoricsUtils.factorial((int) (sideNodeNumber*sideNodeNumber))/(CombinatoricsUtils.factorial((int) ((sideNodeNumber*sideNodeNumber) - 2)) * CombinatoricsUtils.factorial(2)));
        }
        int numberOfRepeatableMembers = (int) (2 * (CombinatoricsUtils.factorial((int) sideNodeNumber)/(CombinatoricsUtils.factorial((int) (sideNodeNumber - 2)) * CombinatoricsUtils.factorial(2))));
        int numVariables = totalNumberOfMembers - numberOfRepeatableMembers;

        int internalPopulationSize = 90;
        int internalMaxEvaluations = 270;

        AbstractProblem metamaterialProblem;
        if (arteryProblem) {
            metamaterialProblem = new ModifiedArteryProblem(integerWeights, constraintNames.length, modelChoice, numVariables, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, engine, heuristicsConstrained, constraintNames);
        } else {
            metamaterialProblem = new ModifiedEqualStiffnessProblem(integerWeights, constraintNames.length, modelChoice, numVariables, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, engine, heuristicsConstrained, constraintNames);
        }

        String[] variableNames = new String[metamaterialProblem.getNumberOfVariables()];
        String[] objectiveNames = new String[metamaterialProblem.getNumberOfObjectives()]; // Attributes names for the unpenalized objectives as recorded in solutions

        for (int i = 0; i < variableNames.length; i++) {
            variableNames[i] = "Variable" + i;
        }

        for (int i = 0; i < objectiveNames.length; i++) {
            objectiveNames[i] = "TrueObjective" + (i+1);
        }

        DominanceComparator internalComparator = new ParetoObjectiveComparator(); // Constraints are now incorporated through interior penalty

        // Internal Operators
        double crossoverProbability = 1.0;
        double mutationProbability = 1.0/numVariables;
        Variation crossover = new OnePointCrossover(crossoverProbability);
        Variation mutation = new BitFlip(mutationProbability);
        Selection internalSelection = new TournamentSelection(2, internalComparator);
        Variation internalVariation;

        double[] epsilonDouble = new double[]{0.0001, 0.0001};
        EpsilonBoxDominanceArchive internalArchive;

        // Initialize MOEA
        int coevolutionPopulationSize = 5;
        int coevolutionMaxEvaluations = 50;

        // Initialize Coevolutionary Problem class
        double[] internalObjectivesMinimum = new double[]{-1, 0}; // Both metamaterial problems involve maximizing the first objective and minimizing the second
        double[] internalObjectivesMaximum = new double[]{0, 1};
        Hypervolume internalHypervolume = new Hypervolume(metamaterialProblem, internalObjectivesMinimum, internalObjectivesMaximum);

        Population coevolutionPopulation = new Population();

        Variation coevolutionaryCrossover;
        Variation coevolutionaryMutation;
        if (integerWeights) {
            coevolutionaryCrossover = new UniformWeightsCrossover(1.0);
            coevolutionaryMutation = new UniformIntegerWeightsMutation(1.0/constraintNames.length);
        } else {
            coevolutionaryCrossover = new SimulatedBinaryWeightsCrossover(1.0, 0.5);
            coevolutionaryMutation = new PM(1.0/constraintNames.length, 0.5); // PM modifies only the RealVariable decisions
        }

        double[] epsilonBoxDouble = new double[metamaterialProblem.getNumberOfObjectives()];
        Arrays.fill(epsilonBoxDouble, 0.0001);
        EpsilonBoxDominanceArchive coevolutionArchive = new EpsilonBoxDominanceArchive(epsilonBoxDouble);
        DominanceComparator coevolutionComparator;
        if (cMOEA) {
            coevolutionComparator = new ParetoObjectiveComparator();
        } else {
            coevolutionComparator = new LinearObjectiveComparator();
        }

        Variation coevolutionVariation;
        Selection coevolutionSelection = new TournamentSelection(2, coevolutionComparator);

        int numberOfCoevolutionaryObjectives = 1;
        if (cMOEA) {
            numberOfCoevolutionaryObjectives = metamaterialProblem.getNumberOfObjectives();
        }

        Population initialInternalPopulation;
        AbstractProblem coevolutionaryProblem;
        Initialization coevolutionInitialization;
        Algorithm coevolutionMOEA;

        String moeaName = "";
        switch (moeaChoice) {
            case 1: // Epsilon-MOEA
                moeaName = "EpsilonMOEA_";
                break;
            case 2: // MOEA-D
                moeaName = "MOEAD_";
                break;
            case 3: // IBEA
                moeaName = "IBEA_";
                break;
            default:
                System.out.println("Invalid model choice");
                break;
        }

        for (int n = 0; n < numRuns; n++) {

            // Initialize random initial internal population and evaluate (to be used for initial fitness calculation)
            List<Solution> internalInitialSolutions = new ArrayList<>();
            for (int i = 0; i < internalPopulationSize; i++) {
                Solution initialSolution = metamaterialProblem.newSolution();
                //for (int j = 0; j < initialSolution.getNumberOfVariables(); j++) {
                //initialSolution.getVariable(j).randomize();
                //}
                metamaterialProblem.evaluate(initialSolution);
                internalInitialSolutions.add(initialSolution);
            }
            initialInternalPopulation = new Population(internalInitialSolutions);

            Result result;
            if (weightOfWeights) {
                result = new Result(saveDir, constraintNames.length+1);
            } else {
                result = new Result(saveDir, constraintNames.length);
            }

            internalArchive = new EpsilonBoxDominanceArchive(epsilonDouble);

            internalVariation = new CompoundVariation(crossover, mutation);

            coevolutionaryProblem = new CoevolutionaryProblem(metamaterialProblem, saveDir, coevolutionPopulationSize, internalMaxEvaluations, initialInternalPopulation, internalArchive, internalComparator, internalSelection, internalVariation, internalHypervolume, cMOEA, evolveInternalPopulation, integerWeights, weightOfWeights, variableNames, objectiveNames, constraintNames, numberOfCoevolutionaryObjectives, result, false, moeaChoice, n, moeaName, problemChoice);

            // NOTE: Injected initialization is used since RandomInitialization will randomize all variable in a new solution (including the last binary variable
            // signifying whether to update initial internal population or not, which we don't want. The initial population of weights must use the same initial internal population
            List<Solution> initialCoevolutionSolutions = new ArrayList<>();

            // Initialize solution of all zero weights and add to the initial population
            Solution zeroSolution = coevolutionaryProblem.newSolution();
            double[] zeroSolutionWeights = new double[zeroSolution.getNumberOfVariables()-1];
            if (weightOfWeights) {
                RealVariable zeroWeightVariable = new RealVariable(0.0, 0.0, 1.0);
                zeroSolution.setVariable(0, zeroWeightVariable);
            } else {
                if (!integerWeights) {
                    Arrays.fill(zeroSolutionWeights, -3); // Real weights = 10^(decisions)
                }
                EncodingUtils.setReal(zeroSolution, 0, zeroSolution.getNumberOfVariables()-1, zeroSolutionWeights);
            }

            initialCoevolutionSolutions.add(zeroSolution);

            // Add other randomly generated solutions
            for (int i = 0; i < coevolutionPopulationSize-1; i++) {
                initialCoevolutionSolutions.add(coevolutionaryProblem.newSolution()); // This new solution by default generates a random set of weights with the internal population not to be updated
            }
            coevolutionInitialization = new InjectedInitialization(coevolutionaryProblem, coevolutionPopulationSize, initialCoevolutionSolutions);

            coevolutionVariation = new CompoundVariation(coevolutionaryCrossover, coevolutionaryMutation);

            if (cMOEA) {
                coevolutionMOEA = null;
                switch (moeaChoice) { // moeaChoice can be switched out for a separate variable to decouple the internal and external MOEA choices
                    case 1:
                        coevolutionMOEA = new EpsilonMOEA(coevolutionaryProblem, coevolutionPopulation, coevolutionArchive, coevolutionSelection, coevolutionVariation, coevolutionInitialization, coevolutionComparator);
                        break;
                    case 2:
                        int neighbourhoodSize = 5; // Number of neighbourhood problems used for mating
                        double delta = 1; // probability of using only neighbourhood problems for mating instead of entire population
                        double eta = 1; // maximum number of population slots a new solution will replace
                        int updateUtility = 2; // frequency (generations) in which utility values are updated
                        coevolutionMOEA = new MOEAD(coevolutionaryProblem, neighbourhoodSize, coevolutionInitialization, coevolutionVariation, delta, eta, updateUtility);
                        break;
                    case 3:
                        ConstrainedIndicatorFitnessEvaluator ife = new ConstrainedIndicatorFitnessEvaluator(coevolutionaryProblem, new ParetoDominanceComparator());
                        coevolutionMOEA = new IBEA(coevolutionaryProblem, coevolutionArchive, coevolutionInitialization, coevolutionVariation, ife);
                        break;
                    default:
                        System.out.println("Invalid MOEA choice");
                        break;
                }
            } else {
                coevolutionMOEA = new GeneticAlgorithm(coevolutionaryProblem, (AggregateObjectiveComparator) coevolutionComparator, coevolutionInitialization, coevolutionSelection, coevolutionVariation);
            }
            ecs.submit(new CoevolutionaryTestSearch(coevolutionMOEA, coevolutionMaxEvaluations, numberOfCoevolutionaryObjectives, result, coevolutionComparator, cMOEA, evolveInternalPopulation, integerWeights, weightOfWeights, periodicZeroInjection, n, moeaName, problemChoice));
        }

        for (int i = 0; i < numRuns; i++) {
            try {
                ecs.take().get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        pool.shutdown();
        engine.close();

    }
}
