package seakers.utils;

import org.moeaframework.core.*;
import org.moeaframework.core.comparator.AggregateConstraintComparator;
import org.moeaframework.core.fitness.IndicatorFitnessEvaluator;
import org.moeaframework.core.indicator.Hypervolume;

public class HypervolumeDifferenceFitnessEvaluator extends IndicatorFitnessEvaluator {

    private final double[] objectivesMaximum;
    private final double[] objectivesMinimum;
    private AggregateConstraintComparator constraintComparator;
    //private Hypervolume hypervolume;
    private double maxAbsIndicatorValue;
    private double[][] fitcomp;

    public HypervolumeDifferenceFitnessEvaluator(Problem problem, double[] objectivesMinimum, double[] objectivesMaximum) {
        super(problem);
        this.objectivesMinimum = objectivesMinimum;
        this.objectivesMaximum = objectivesMaximum;
        this.constraintComparator = new AggregateConstraintComparator();
        //this.hypervolume = new Hypervolume(problem, objectivesMinimum, objectivesMaximum);
    }

    @Override
    protected double calculateIndicator(Solution solution, Solution solution1) {
        double constraintComparison = this.constraintComparator.compare(solution, solution1);

        if (constraintComparison == 0) {
            //// The inbuilt Hypervolume function does not compute hypervolume for a single solution
            //NondominatedPopulation solutionPopulation = new NondominatedPopulation();
            //solutionPopulation.add(solution);

            //NondominatedPopulation solutionPopulation1 = new NondominatedPopulation();
            //solutionPopulation1.add(solution1);

            //double solutionHypervolume = this.hypervolume.evaluate(solutionPopulation);
            //double solutionHypervolume1 = this.hypervolume.evaluate(solutionPopulation1);

            double solutionHypervolume = computeHypervolumeSingleSolution(solution);
            double solutionHypervolume1 = computeHypervolumeSingleSolution(solution1);

            return solutionHypervolume - solutionHypervolume1;
        } else {
            return -constraintComparison; // constraintComparison = 1 if constraint violation of solution < constraint violation of solution1
            // = -1 vice versa
        }
    }

    @Override
    public boolean areLargerValuesPreferred() {
        return false;
    }

    @Override
    public void evaluate(Population population) {
        //ConstraintAgnosticNormalizer normalizer = new ConstraintAgnosticNormalizer(this.problem, population, this.delta);
        //Population normalizedPopulation = normalizer.normalize(population);
        this.fitcomp = new double[population.size()][population.size()];
        this.maxAbsIndicatorValue = -1.0D / 0.0;

        int i;
        for(i = 0; i < population.size(); ++i) {
            for(int j = 0; j < population.size(); ++j) {
                //this.fitcomp[i][j] = this.calculateIndicator(normalizedPopulation.get(i), normalizedPopulation.get(j));
                this.fitcomp[i][j] = this.calculateIndicator(population.get(i), population.get(j));
                if (Math.abs(this.fitcomp[i][j]) > this.maxAbsIndicatorValue) {
                    this.maxAbsIndicatorValue = Math.abs(this.fitcomp[i][j]);
                }
            }
        }

        for(i = 0; i < population.size(); ++i) {
            double sum = 0.0D;

            for(int j = 0; j < population.size(); ++j) {
                if (i != j) {
                    sum += Math.exp(-this.fitcomp[j][i] / this.maxAbsIndicatorValue / 0.05D);
                }
            }

            population.get(i).setAttribute("fitness", sum);
        }

    }

    @Override
    public void removeAndUpdate(Population population, int removeIndex) {
        if (this.fitcomp == null) {
            throw new FrameworkException("evaluate must be called first");
        } else {
            int i;
            for(i = 0; i < population.size(); ++i) {
                if (i != removeIndex) {
                    Solution solution = population.get(i);
                    double fitness = (Double)solution.getAttribute("fitness");
                    fitness -= Math.exp(-this.fitcomp[removeIndex][i] / this.maxAbsIndicatorValue / 0.05D);
                    solution.setAttribute("fitness", fitness);
                }
            }

            for(i = 0; i < population.size(); ++i) {
                for(int j = removeIndex + 1; j < population.size(); ++j) {
                    this.fitcomp[i][j - 1] = this.fitcomp[i][j];
                }

                if (i > removeIndex) {
                    this.fitcomp[i - 1] = this.fitcomp[i];
                }
            }

            population.remove(removeIndex);
        }
    }

    /**
     * Compute approxiamted hypervolume covered by a solution wrt to the anti-utopia point (assuming that all objectives are to be minimized)
     *
     * @param solution
     * @return hypervolume
     */
    private double computeHypervolumeSingleSolution(Solution solution) {
        double hypervolume = 1;
        for (int i = 0; i < solution.getNumberOfObjectives(); i++) {
            hypervolume *= ((objectivesMaximum[i] - solution.getObjective(i))/(objectivesMaximum[i] - objectivesMinimum[i]));
        }
        return hypervolume;
    }
}
