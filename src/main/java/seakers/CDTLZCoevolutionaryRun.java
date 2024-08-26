package seakers;

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
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.core.operator.UniformCrossover;
import org.moeaframework.core.operator.real.PM;
import org.moeaframework.core.operator.real.SBX;
import org.moeaframework.core.operator.real.UM;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.problem.CDTLZ.*;
import seakers.operators.SimulatedBinaryWeightsCrossover;
import seakers.operators.UniformIntegerWeightsMutation;
import seakers.operators.UniformWeightsCrossover;
import seakers.problems.C_DTLZ.ModifiedC_DTLZProblem;
import seakers.problems.CoevolutionaryProblem;
import seakers.utils.ConstrainedIndicatorFitnessEvaluator;
import seakers.utils.HypervolumeDifferenceFitnessEvaluator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

public class CDTLZCoevolutionaryRun {

    public static ExecutorService pool;
    public static CompletionService<Algorithm> cs;

    public static void main(String[] args) {

        // Save location
        //String saveDir = System.getProperty("user.dir") + File.separator + "results"; // File not found error!
        String saveDir = "C:\\SEAK Lab\\SEAK Lab Github\\Coevolution-based Constraint Satisfaction\\results";

        // Problem parameters
        int numCPU = 4;
        int numRuns = 30; // comment if lines 73 and 74 are uncommented

        pool = Executors.newFixedThreadPool(numCPU);
        cs = new ExecutorCompletionService<>(pool);

        double crossoverProbability = 1.0;

        boolean cMOEA = false; // if the coevolutionary optimization is single or multi objective
        boolean evolveInternalPopulation = true; // if initial internal population for subsequent heuristic weights should be updated or not
        boolean integerWeights = false; // if formulation for the heuristic weights should be integer or real
        boolean weightOfWeights = false; // if an additional weight multiplicative parameter design decision for the outer GA is used
        boolean periodicZeroInjection = false; // if zero solution is added for evaluation to population at different NFE

        int moeaChoice = 1; // choice of internal moea algorithm: 1 -> epsilon-MOEA, 2 -> MOEA-D. 3 -> IBEA

        int internalPopulationSize = 0;
        int internalMaxEvaluations = 0;

        // Define problem
        AbstractProblem CDTLZProblem = null; // Change problem as required
        int problemChoice = 5; // 1 -> C1-DTLZ1, 2 -> C1-DTLZ3, 3-> C2-DTLZ2, 4 -> C3-DTLZ1, 5 -> C3-DTLZ4
        int numberOfObjectives = 6; // only applicable if problem choice = 0
        String problemName = "";
        double[] objectiveNormalizations = null; // denominators for each objective to normalize the values to between 0 and 1 (minimum value for all objectives is 0)
        double[] constraintNormalizations = null; // denominators for each constraint to normalize the values to between 0 and 1 (minimum value for all constraints is 0)

        switch (problemChoice) {
            case 1:
                CDTLZProblem = new C1_DTLZ1(numberOfObjectives);
                problemName = "c1_dtlz1_" + numberOfObjectives + "_";
                internalPopulationSize = 30*numberOfObjectives;
                internalMaxEvaluations = 100*numberOfObjectives;
                objectiveNormalizations = new double[]{371, 455, 510};
                constraintNormalizations = new double[]{946};
                if (numberOfObjectives == 6) {
                    objectiveNormalizations = new double[]{221, 201, 296, 418, 499, 464};
                    constraintNormalizations = new double[]{895};
                } else if (numberOfObjectives == 12) {
                    objectiveNormalizations = new double[]{32, 26, 38, 74, 120, 143, 166, 208, 349, 348, 483, 481};
                    constraintNormalizations = new double[]{879};
                }

                break;
            case 2:
                CDTLZProblem = new C1_DTLZ3(numberOfObjectives);
                problemName = "c1_dtlz3_" + numberOfObjectives + "_";
                internalPopulationSize = 60*numberOfObjectives;
                internalMaxEvaluations = 200*numberOfObjectives;
                objectiveNormalizations = new double[]{1720, 1659, 1729};
                constraintNormalizations = new double[]{1057};
                if (numberOfObjectives == 6) {
                    objectiveNormalizations = new double[]{1593, 1761, 1680, 1673, 1741, 1766};
                    constraintNormalizations = new double[]{1057};
                } else if (numberOfObjectives == 12) {
                    objectiveNormalizations = new double[]{1413, 1396, 1520, 1685, 1740, 1844, 1813, 1811, 1787, 1834, 1841, 1754};
                    constraintNormalizations = new double[]{1057};
                }

                break;
            case 3:
                CDTLZProblem = new C2_DTLZ2(numberOfObjectives);
                problemName = "c2_dtlz2_" + numberOfObjectives + "_";
                internalPopulationSize = 60*numberOfObjectives;
                internalMaxEvaluations = 200*numberOfObjectives;
                objectiveNormalizations = new double[]{2.4, 2.5, 2.7};
                constraintNormalizations = new double[]{3.5};
                if (numberOfObjectives == 6) {
                    objectiveNormalizations = new double[]{2, 2.1, 2.5, 2.4, 2.5, 2.6};
                    constraintNormalizations = new double[]{4.1};
                } else if (numberOfObjectives == 12) {
                    objectiveNormalizations = new double[]{1.4, 1.2, 1.3, 1.2, 1.7, 1.7, 1.9, 2.2, 2.5, 2.3, 2.6, 2.8};
                    constraintNormalizations = new double[]{4.8};
                }

                break;
            case 4:
                CDTLZProblem = new C3_DTLZ1(numberOfObjectives);
                problemName = "c3_dtlz1_" + numberOfObjectives + "_";
                internalPopulationSize = 90*numberOfObjectives;
                internalMaxEvaluations = 300*numberOfObjectives;
                    objectiveNormalizations = new double[]{422, 417, 482};
                constraintNormalizations = new double[]{1, 1, 1};
                if (numberOfObjectives == 6) {
                    objectiveNormalizations = new double[]{403, 388, 454, 452, 482, 473};
                    constraintNormalizations = new double[]{1, 1, 1, 1, 1, 1};
                } else if (numberOfObjectives == 12) {
                    objectiveNormalizations = new double[]{182, 172, 214, 263, 325, 398, 415, 459, 442, 478, 485, 506};
                    constraintNormalizations = new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
                }

                break;
            case 5:
                CDTLZProblem = new C3_DTLZ4(numberOfObjectives);
                problemName = "c3_dtlz4_" + numberOfObjectives + "_";
                internalPopulationSize = 90*numberOfObjectives;
                internalMaxEvaluations = 300*numberOfObjectives;
                objectiveNormalizations = new double[]{2.8, 2.6, 2.6};
                constraintNormalizations = new double[]{0.7, 0.7, 0.5};
                if (numberOfObjectives == 6) {
                    objectiveNormalizations = new double[]{2.9, 2.6, 2.6, 2.6, 2.6, 2.6};
                    constraintNormalizations = new double[]{0.7, 0.6, 0.6, 0.6, 0.6, 0.6};
                } else if (numberOfObjectives == 12) {
                    objectiveNormalizations = new double[]{2.9, 2.7, 2.7, 2.7, 2.7, 2.8, 2.8, 2.7, 2.7, 2.8, 2.8, 2.7};
                    constraintNormalizations = new double[]{0.8, 0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.7, 0.6, 0.5, 0.6, 0.5};
                }

                break;
            default:
                System.out.println("Invalid choice of C_DTLZ problem");
                break;
        }
        AbstractProblem internalProblem = new ModifiedC_DTLZProblem(CDTLZProblem, CDTLZProblem.getNumberOfConstraints(), integerWeights, objectiveNormalizations, constraintNormalizations);

        String[] variableNames = new String[CDTLZProblem.getNumberOfVariables()];
        String[] objectiveNames = new String[CDTLZProblem.getNumberOfObjectives()];
        String[] constraintNames = new String[CDTLZProblem.getNumberOfConstraints()];

        for (int i = 0; i < variableNames.length; i++) {
            variableNames[i] = "Variable" + i;
        }

        for (int i = 0; i < objectiveNames.length; i++) {
            objectiveNames[i] = "TrueObjective" + (i+1);
        }

        for (int i = 0; i < constraintNames.length; i++) {
            constraintNames[i] = "Constraint" + i;
        }

        double[] epsilonBox = new double[CDTLZProblem.getNumberOfObjectives()];
        Arrays.fill(epsilonBox, 0.0001);

        double mutationProbability = 1.0/CDTLZProblem.getNumberOfVariables();
        Variation crossover = new SBX(crossoverProbability, 15); // taken from original DTLZ paper
        Variation mutation = new PM(mutationProbability, 20); // taken from original DTLZ paper

        DominanceComparator internalComparator = new ParetoObjectiveComparator(); // Constraints are now incorporated through interior penalty

        Selection internalSelection = new TournamentSelection(2, internalComparator);
        Variation internalVariation;

        double[] epsilonDouble = new double[]{0.0001, 0.0001};
        EpsilonBoxDominanceArchive internalArchive;

        // Initialize MOEA
        int coevolutionPopulationSize = 6;
        int coevolutionMaxEvaluations = 60;

        // Initialize Coevolutionary Problem class
        double[] internalObjectivesMinimum = new double[internalProblem.getNumberOfObjectives()]; // Both metamaterial problems involve maximizing the first objective and minimizing the second
        double[] internalObjectivesMaximum = new double[internalProblem.getNumberOfObjectives()];
        Arrays.fill(internalObjectivesMinimum, 0);
        Arrays.fill(internalObjectivesMaximum, 1);
        Hypervolume internalHypervolume = new Hypervolume(internalProblem, internalObjectivesMinimum, internalObjectivesMaximum);

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

        double[] epsilonBoxDouble = new double[CDTLZProblem.getNumberOfObjectives()];
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
            numberOfCoevolutionaryObjectives = internalProblem.getNumberOfObjectives();
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
                Solution initialSolution = internalProblem.newSolution();
                for (int j = 0; j < initialSolution.getNumberOfVariables(); j++) {
                    initialSolution.getVariable(j).randomize();
                }
                //initialSolution.setAttribute("AlreadyEvaluated", false);
                //problem.evaluate(initialSolution);
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

            coevolutionaryProblem = new CoevolutionaryProblem(internalProblem, saveDir, coevolutionPopulationSize, internalMaxEvaluations, initialInternalPopulation, internalArchive, internalComparator, internalSelection, internalVariation, internalHypervolume, cMOEA, evolveInternalPopulation, integerWeights, weightOfWeights, variableNames, objectiveNames, constraintNames, numberOfCoevolutionaryObjectives, result, true, moeaChoice, n, moeaName, problemName);

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
                        coevolutionMOEA = new EpsilonMOEA(coevolutionaryProblem, coevolutionPopulation, coevolutionArchive, coevolutionSelection, coevolutionVariation, coevolutionInitialization, coevolutionComparator);;
                        break;
                    case 2:
                        int neighbourhoodSize = 5; // Number of neighbourhood problems used for mating
                        double delta = 1; // probability of using only neighbourhood problems for mating instead of entire population
                        double eta = 1; // maximum number of population slots a new solution will replace
                        int updateUtility = 2; // frequency (generations) in which utility values are updated
                        coevolutionMOEA = new MOEAD(coevolutionaryProblem, neighbourhoodSize, coevolutionInitialization, coevolutionVariation, delta, eta, updateUtility);
                        break;
                    case 3:

                        //double IBEADelta = 0.0;
                        //if (numberOfObjectives > 10) {
                            //IBEADelta = 10;
                        //}
                        //ConstrainedIndicatorFitnessEvaluator ife = new ConstrainedIndicatorFitnessEvaluator(coevolutionaryProblem, new ParetoObjectiveComparator(), IBEADelta);

                        double[] normalizedObjectivesMaximum = new double[coevolutionaryProblem.getNumberOfObjectives()];
                        double[] normalizedObjectivesMinimum = new double[coevolutionaryProblem.getNumberOfObjectives()];
                        Arrays.fill(normalizedObjectivesMaximum, 1.0);
                        Arrays.fill(normalizedObjectivesMinimum, 0.0);

                        HypervolumeDifferenceFitnessEvaluator ife = new HypervolumeDifferenceFitnessEvaluator(coevolutionaryProblem, normalizedObjectivesMinimum, normalizedObjectivesMaximum);

                        coevolutionMOEA = new IBEA(coevolutionaryProblem, coevolutionArchive, coevolutionInitialization, coevolutionVariation, ife);
                        break;
                    default:
                        System.out.println("Invalid MOEA choice");
                        break;
                }
            } else {
                coevolutionMOEA = new GeneticAlgorithm(coevolutionaryProblem, (AggregateObjectiveComparator) coevolutionComparator, coevolutionInitialization, coevolutionSelection, coevolutionVariation);
            }
            cs.submit(new CoevolutionaryTestSearch(coevolutionMOEA, coevolutionMaxEvaluations, numberOfCoevolutionaryObjectives, result, coevolutionComparator, cMOEA, evolveInternalPopulation, integerWeights, weightOfWeights, periodicZeroInjection, n, moeaName, problemName));
        }

        for (int i = 0; i < numRuns; i++) {
            try {
                cs.take().get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        pool.shutdown();

    }
}
