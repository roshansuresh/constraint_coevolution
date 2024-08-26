package seakers.utils;

import org.moeaframework.core.FrameworkException;
import org.moeaframework.core.Population;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Solution;
import org.moeaframework.core.comparator.DominanceComparator;
import org.moeaframework.core.fitness.IndicatorFitnessEvaluator;

public class ConstrainedIndicatorFitnessEvaluator extends IndicatorFitnessEvaluator {

    private final DominanceComparator comp;
    private Problem problem;
    private static final double kappa = 0.05D;
    private double delta = 0.0D;
    private double maxAbsIndicatorValue;
    private double[][] fitcomp;

    public ConstrainedIndicatorFitnessEvaluator(Problem problem, DominanceComparator comp) {
        super(problem);
        this.problem = problem;
        this.comp = comp;
    }

    public ConstrainedIndicatorFitnessEvaluator(Problem problem, DominanceComparator comp, double delta) {
        super(problem);
        this.problem = problem;
        this.comp = comp;
        this.delta = delta;
    }

    @Override
    protected double calculateIndicator(Solution solution, Solution solution1) {
        //DominanceComparator comp = new ParetoDominanceComparator();
        return this.comp.compare(solution, solution1);
    }

    @Override
    public boolean areLargerValuesPreferred() {
        return false;
    }

    @Override
    public void evaluate(Population population) {
        ConstraintAgnosticNormalizer normalizer = new ConstraintAgnosticNormalizer(this.problem, population, this.delta);
        Population normalizedPopulation = normalizer.normalize(population);
        this.fitcomp = new double[population.size()][population.size()];
        this.maxAbsIndicatorValue = -1.0D / 0.0;

        int i;
        for(i = 0; i < population.size(); ++i) {
            for(int j = 0; j < population.size(); ++j) {
                this.fitcomp[i][j] = this.calculateIndicator(normalizedPopulation.get(i), normalizedPopulation.get(j));
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

}
