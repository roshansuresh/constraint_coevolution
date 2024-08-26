package seakers;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.algorithm.EpsilonMOEA;
import org.moeaframework.algorithm.IBEA;
import org.moeaframework.algorithm.MOEAD;
import org.moeaframework.core.*;
import org.moeaframework.core.comparator.*;
import org.moeaframework.core.operator.*;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.operator.real.PM;
import org.moeaframework.core.operator.real.SBX;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.problem.CDTLZ.*;
import seakers.problems.C_DTLZ.ModifiedC_DTLZProblem;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;
import seakers.utils.ConstrainedIndicatorFitnessEvaluator;
import seakers.utils.HypervolumeDifferenceFitnessEvaluator;
import seakers.utils.SynchronizedMersenneTwister;

import java.util.Arrays;
import java.util.concurrent.*;

public class MOEARun {

    public static ExecutorService pool;
    public static CompletionService<Algorithm> cs;
    /**
     * Matlab Engine for function evaluation
     */
    private static MatlabEngine engine;

    public static void main (String[] args) throws InterruptedException, ExecutionException {

        // Save location
        //String saveDir = System.getProperty("user.dir") + File.separator + "results"; // File not found error!
        String saveDir = "C:\\SEAK Lab\\SEAK Lab Github\\Coevolution-based Constraint Satisfaction\\results";

        // Problem parameters
        int populationSize = 0; // set subsequently based on problem
        int maximumEvaluations = 0; // set subsequently based on problem

        int numCPU = 4;
        int numRuns = 30; // comment if lines 73 and 74 are uncommented

        pool = Executors.newFixedThreadPool(numCPU);
        cs = new ExecutorCompletionService<>(pool);

        double crossoverProbability = 1.0;

        int moeaChoice = 2; // choice of moea algorithm: 1 -> epsilon-MOEA, 2 -> MOEA-D. 3 -> IBEA

        // Test problem
        int problemChoice = 0; // Problem choice = 0 -> Cx_DTLZy, 1 -> Either Metamaterial problem
        int cdtlzProblemChoice = 4; // 1 -> C1-DTLZ1, 2 -> C1-DTLZ3, 3-> C2-DTLZ2, 4 -> C3-DTLZ1, 5 -> C3-DTLZ4 (only applicable if problem choice = 0)
        int numberOfObjectives = 6; // only applicable if problem choice = 0
        boolean arteryProblem = true; // Only useful if problemChoice is 1

        // Defined for all problems
        boolean[][] heuristicsConstrained = new boolean[0][]; // Only for the metamaterial problems
        String[] variableNames = new String[0];
        String[] objectiveNames = new String[0];
        String[] constraintNames = new String[0];
        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        double[] epsilonBox = new double[0];
        Initialization initialization = null;
        Variation crossover = null;
        Variation mutation = null;

        double mutationProbability = 0.0;

        PRNG.setRandom(new SynchronizedMersenneTwister());

        AbstractProblem problem = null;
        String problemName = "";
        double[] objectiveNormalizations = null; // denominators for each objective to normalize the values to between 0 and 1 (minimum value for all objectives is 0)
        double[] constraintNormalizations = null; // denominators for each constraint to normalize the values to between 0 and 1 (minimum value for all constraints is 0)

        switch (problemChoice) {

            case 0:
                AbstractProblem cdtlzProblem = null;
                switch (cdtlzProblemChoice) {
                    case 1:
                        cdtlzProblem = new C1_DTLZ1(numberOfObjectives);
                        problemName = "c1_dtlz1_" + numberOfObjectives + "_";
                        populationSize = 30*numberOfObjectives;
                        maximumEvaluations = 6000*numberOfObjectives;

                        objectiveNormalizations = new double[]{371, 455, 510};
                        constraintNormalizations = new double[]{946};
                        if (numberOfObjectives == 6) {
                            objectiveNormalizations = new double[]{221, 201, 296, 418, 499, 464};
                            constraintNormalizations = new double[]{895};
                        } else if (numberOfObjectives == 12) {
                            objectiveNormalizations = new double[]{32, 26, 38, 74, 120, 143, 166, 208, 349, 348, 483, 481};
                            constraintNormalizations = new double[]{879};
                        }

                        //// No normalization to get objective and constraint bounds
                        //objectiveNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(objectiveNormalizations, 1);
                        //constraintNormalizations = new double[]{1};

                        break;
                    case 2:
                        cdtlzProblem = new C1_DTLZ3(numberOfObjectives);
                        problemName = "c1_dtlz3_" + numberOfObjectives + "_";
                        populationSize = 60*numberOfObjectives;
                        maximumEvaluations = 12000*numberOfObjectives;

                        objectiveNormalizations = new double[]{1720, 1659, 1729};
                        constraintNormalizations = new double[]{1057};
                        if (numberOfObjectives == 6) {
                            objectiveNormalizations = new double[]{1593, 1761, 1680, 1673, 1741, 1766};
                            constraintNormalizations = new double[]{1057};
                        } else if (numberOfObjectives == 12) {
                            objectiveNormalizations = new double[]{1413, 1396, 1520, 1685, 1740, 1844, 1813, 1811, 1787, 1834, 1841, 1754};
                            constraintNormalizations = new double[]{1057};
                        }

                        //// No normalization to get objective and constraint bounds
                        //objectiveNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(objectiveNormalizations, 1);
                        //constraintNormalizations = new double[]{1};

                        break;
                    case 3:
                        cdtlzProblem = new C2_DTLZ2(numberOfObjectives);
                        problemName = "c2_dtlz2_" + numberOfObjectives + "_";
                        populationSize = 60*numberOfObjectives;
                        maximumEvaluations = 12000*numberOfObjectives;

                        objectiveNormalizations = new double[]{2.4, 2.5, 2.7};
                        constraintNormalizations = new double[]{3.5};
                        if (numberOfObjectives == 6) {
                            objectiveNormalizations = new double[]{2, 2.1, 2.5, 2.4, 2.5, 2.6};
                            constraintNormalizations = new double[]{4.1};
                        } else if (numberOfObjectives == 12) {
                            objectiveNormalizations = new double[]{1.4, 1.2, 1.3, 1.2, 1.7, 1.7, 1.9, 2.2, 2.5, 2.3, 2.6, 2.8};
                            constraintNormalizations = new double[]{4.8};
                        }

                        //// No normalization to get objective and constraint bounds
                        //objectiveNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(objectiveNormalizations, 1);
                        //constraintNormalizations = new double[]{1};

                        break;
                    case 4:
                        cdtlzProblem = new C3_DTLZ1(numberOfObjectives);
                        problemName = "c3_dtlz1_" + numberOfObjectives + "_";
                        populationSize = 90*numberOfObjectives;
                        maximumEvaluations = 18000*numberOfObjectives;

                        objectiveNormalizations = new double[]{422, 417, 482};
                        constraintNormalizations = new double[]{1, 1, 1};
                        if (numberOfObjectives == 6) {
                            objectiveNormalizations = new double[]{403, 388, 454, 452, 482, 473};
                            constraintNormalizations = new double[]{1, 1, 1, 1, 1, 1};
                        } else if (numberOfObjectives == 12) {
                            objectiveNormalizations = new double[]{182, 172, 214, 263, 325, 398, 415, 459, 442, 478, 485, 506};
                            constraintNormalizations = new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
                        }

                        //// No normalization to get objective and constraint bounds
                        //objectiveNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(objectiveNormalizations, 1);
                        //constraintNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(constraintNormalizations, 1);

                        break;
                    case 5:
                        cdtlzProblem = new C3_DTLZ4(numberOfObjectives);
                        problemName = "c3_dtlz4_" + numberOfObjectives + "_";
                        populationSize = 90*numberOfObjectives;
                        maximumEvaluations = 18000*numberOfObjectives;

                        objectiveNormalizations = new double[]{2.8, 2.6, 2.6};
                        constraintNormalizations = new double[]{0.7, 0.7, 0.5};
                        if (numberOfObjectives == 6) {
                            objectiveNormalizations = new double[]{2.9, 2.6, 2.6, 2.6, 2.6, 2.6};
                            constraintNormalizations = new double[]{0.7, 0.6, 0.6, 0.6, 0.6, 0.6};
                        } else if (numberOfObjectives == 12) {
                            objectiveNormalizations = new double[]{2.9, 2.7, 2.7, 2.7, 2.7, 2.8, 2.8, 2.7, 2.7, 2.8, 2.8, 2.7};
                            constraintNormalizations = new double[]{0.8, 0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.7, 0.6, 0.5, 0.6, 0.5};
                        }

                        //// No normalization to get objective and constraint bounds
                        //objectiveNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(objectiveNormalizations, 1);
                        //constraintNormalizations = new double[numberOfObjectives];
                        //Arrays.fill(constraintNormalizations, 1);

                        break;
                    default:
                        System.out.println("Invalid choice of C_DTLZ problem");
                        break;
                }

                variableNames = new String[cdtlzProblem.getNumberOfVariables()];
                objectiveNames = new String[cdtlzProblem.getNumberOfObjectives()];
                constraintNames = new String[cdtlzProblem.getNumberOfConstraints()];

                for (int i = 0; i < variableNames.length; i++) {
                    variableNames[i] = "Variable" + i;
                }

                for (int i = 0; i < objectiveNames.length; i++) {
                    objectiveNames[i] = "TrueObjective" + (i+1);
                }

                for (int i = 0; i < constraintNames.length; i++) {
                    constraintNames[i] = "Constraint" + i;
                }

                epsilonBox = new double[cdtlzProblem.getNumberOfObjectives()];
                Arrays.fill(epsilonBox, 0.0001);

                mutationProbability = 1.0/cdtlzProblem.getNumberOfVariables();
                crossover = new SBX(crossoverProbability, 15); // taken from original DTLZ paper
                mutation = new PM(mutationProbability, 20); // taken from original DTLZ paper

                problem = new ModifiedC_DTLZProblem(cdtlzProblem, cdtlzProblem.getNumberOfConstraints(), false, objectiveNormalizations, constraintNormalizations);

                break;

            case 1:
                engine = MatlabEngine.startMatlab();

                String matlabScriptsLocation = "C:\\SEAK Lab\\SEAK Lab Github\\KD3M3\\Truss_AOS";
                engine.feval("addpath", matlabScriptsLocation); // Add location of MATLAB scripts used to compute objectives, constraints and heuristics to MATLAB's search path

                /**
                 * modelChoice = 0 --> Fibre Stiffness Model
                 *             = 1 --> Truss Stiffness Model
                 *             = 2 --> Beam Model
                 */
                int modelChoice = 1; // Fibre stiffness model cannot be used for the artery problem

                double targetStiffnessRatio = 1;
                if (arteryProblem) {
                    targetStiffnessRatio = 0.421;
                }

                populationSize = 90;
                maximumEvaluations = 13500;

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

                heuristicsConstrained = new boolean[4][7];
                for (int i = 0; i < 7; i++) {
                    heuristicsConstrained[0][i] = partialCollapsibilityConstrained[i];
                    heuristicsConstrained[1][i] = nodalPropertiesConstrained[i];
                    heuristicsConstrained[2][i] = orientationConstrained[i];
                    heuristicsConstrained[3][i] = intersectionConstrained[i];
                }

                numberOfHeuristicConstraints = 0;
                numberOfHeuristicObjectives = 0;
                for (int i = 0; i < 4; i++) {
                    if (heuristicsConstrained[i][5]) {
                        numberOfHeuristicConstraints++;
                    }
                    if (heuristicsConstrained[i][4]) {
                        numberOfHeuristicObjectives++;
                    }
                }

                // New dimensions for printable solutions
                double printableRadius = 250e-6; // in m
                double printableSideLength = 10e-3; // in m
                double printableModulus = 1.8162e6; // in Pa
                double sideNodeNumber = 3.0D;
                int nucFactor = 3; // Not used if PBC model is used

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

                double[][] globalNodePositions;
                if (arteryProblem) {
                    problem = new ConstantRadiusArteryProblem(saveDir, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, engine, heuristicsConstrained);
                    globalNodePositions = ((ConstantRadiusArteryProblem) problem).getNodalConnectivityArray();
                } else {
                    problem = new ConstantRadiusTrussProblem2(saveDir, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, engine, heuristicsConstrained);
                    globalNodePositions = ((ConstantRadiusTrussProblem2) problem).getNodalConnectivityArray();
                }
                variableNames = new String[problem.getNumberOfVariables()];
                objectiveNames = new String[problem.getNumberOfObjectives()];
                constraintNames = new String[problem.getNumberOfConstraints()];

                for (int i = 0; i < variableNames.length; i++) {
                    variableNames[i] = "Variable" + i;
                }

                for (int i = 0; i < objectiveNames.length; i++) {
                    objectiveNames[i] = "TrueObjective" + (i+1);
                }

                if (arteryProblem) {
                    problemName = "artery_";
                    constraintNames = new String[]{"FeasibilityViolation", "ConnectivityViolation"};
                } else {
                    problemName = "equalstiffness_";
                    constraintNames = new String[]{"FeasibilityViolation", "ConnectivityViolation", "StiffnessRatioViolation"};
                }

                epsilonBox = new double[problem.getNumberOfObjectives()];
                Arrays.fill(epsilonBox, 0.0001);

                mutationProbability = 1.0/problem.getNumberOfVariables();
                crossover = new OnePointCrossover(crossoverProbability);
                mutation = new BitFlip(mutationProbability);

                break;

            default:
                System.out.println("No problem chosen");
        }

        DominanceComparator comparator = new ChainedComparator(new AggregateConstraintComparator(), new ParetoObjectiveComparator());

        Variation variation;
        EpsilonBoxDominanceArchive archive;
        Selection selection;

        for (int i = 0; i < numRuns; i++) {
            Population population = new Population();
            initialization = new RandomInitialization(problem, populationSize);
            archive = new EpsilonBoxDominanceArchive(epsilonBox);
            variation = new CompoundVariation(crossover, mutation);
            selection = new TournamentSelection(2, comparator);
            Algorithm moea = null;
            switch (moeaChoice) {
                case 1:
                    moea = new EpsilonMOEA(problem, population, archive, selection, variation, initialization, comparator);
                    break;
                case 2:
                    int neighbourhoodSize = 5; // Number of neighbourhood problems used for mating
                    double delta = 1; // probability of using only neighbourhood problems for mating instead of entire population
                    double eta = 1; // maximum number of population slots a new solution will replace
                    int updateUtility = 5; // frequency (generations) in which utility values are updated
                    moea = new MOEAD(problem, neighbourhoodSize, initialization, variation, delta, eta, updateUtility);
                    break;
                case 3:

                    //double IBEADelta = 0.0;
                    //if (numberOfObjectives > 10) {
                        //IBEADelta = 10;
                    //}
                    //ConstrainedIndicatorFitnessEvaluator ife = new ConstrainedIndicatorFitnessEvaluator(problem, new ParetoDominanceComparator(), IBEADelta);

                    double[] normalizedObjectivesMaximum = new double[problem.getNumberOfObjectives()];
                    double[] normalizedObjectivesMinimum = new double[problem.getNumberOfObjectives()];
                    Arrays.fill(normalizedObjectivesMaximum, 1.0);
                    Arrays.fill(normalizedObjectivesMinimum, 0.0);

                    HypervolumeDifferenceFitnessEvaluator ife = new HypervolumeDifferenceFitnessEvaluator(problem, normalizedObjectivesMinimum, normalizedObjectivesMaximum);

                    moea = new IBEA(problem, archive, initialization, variation, ife);
                    break;
                default:
                    System.out.println("Invalid MOEA choice");
                    break;
            }

            cs.submit(new MOEASearch(moea, saveDir, maximumEvaluations, i, variableNames, objectiveNames, constraintNames, problemName,problemChoice==0, moeaChoice));
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
