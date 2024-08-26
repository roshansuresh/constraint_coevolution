package seakers.utils;

import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Solution;
import org.moeaframework.core.indicator.Normalizer;

import java.util.Arrays;
import java.util.Iterator;

public class ConstraintAgnosticNormalizer {
    private final Problem problem;
    private final double delta;
    private final double[] referencePoint;
    private final double[] minimum;
    private final double[] maximum;

    public ConstraintAgnosticNormalizer(Problem problem, Population population) {
        this.problem = problem;
        this.delta = 0.0D;
        this.referencePoint = null;
        this.minimum = new double[problem.getNumberOfObjectives()];
        this.maximum = new double[problem.getNumberOfObjectives()];
        this.calculateRanges(population);
        this.checkRanges();
    }

    public ConstraintAgnosticNormalizer(Problem problem, Population population, double delta) {
        this.problem = problem;
        this.delta = delta;
        this.referencePoint = null;
        this.minimum = new double[problem.getNumberOfObjectives()];
        this.maximum = new double[problem.getNumberOfObjectives()];
        this.calculateRanges(population);
        this.checkRanges();
    }

    public ConstraintAgnosticNormalizer(Problem problem, Population population, double[] referencePoint) {
        this.problem = problem;
        this.delta = 0.0D;
        this.referencePoint = (double[])referencePoint.clone();
        this.minimum = new double[problem.getNumberOfObjectives()];
        this.maximum = new double[problem.getNumberOfObjectives()];
        this.calculateRanges(population);
        this.checkRanges();
    }

    public ConstraintAgnosticNormalizer(Problem problem, double[] minimum, double[] maximum) {
        this.problem = problem;
        this.delta = 0.0D;
        this.referencePoint = null;
        this.minimum = new double[problem.getNumberOfObjectives()];
        this.maximum = new double[problem.getNumberOfObjectives()];

        for(int j = 0; j < problem.getNumberOfObjectives(); ++j) {
            this.minimum[j] = minimum[j >= minimum.length ? minimum.length - 1 : j];
            this.maximum[j] = maximum[j >= maximum.length ? maximum.length - 1 : j];
        }

        this.checkRanges();
    }

    private void calculateRanges(Population population) {
        if (population.size() < 2) {
            throw new IllegalArgumentException("requires at least two solutions");
        } else {
            for(int j = 0; j < this.problem.getNumberOfObjectives(); ++j) {
                this.minimum[j] = 1.0D/0.0;
                this.maximum[j] = -1.0D/0.0;
            }

            for(int j = 0; j < population.size(); ++j) {
                Solution solution = population.get(j);
                for(int k = 0; k < this.problem.getNumberOfObjectives(); ++k) {
                    this.minimum[k] = Math.min(this.minimum[k], solution.getObjective(k));
                    this.maximum[k] = Math.max(this.maximum[k], solution.getObjective(k));
                }
            }

            if (this.referencePoint != null) {
                for(int j = 0; j < this.problem.getNumberOfObjectives(); ++j) {
                    this.maximum[j] = this.referencePoint[j >= this.referencePoint.length ? this.referencePoint.length - 1 : j];
                }

                System.err.println("Using reference point: " + Arrays.toString(this.maximum));
            } else if (this.delta > 0.0D) {
                for(int j = 0; j < this.problem.getNumberOfObjectives(); ++j) {
                    this.maximum[j] += this.delta * (this.maximum[j] - this.minimum[j]);
                }

                System.err.println("Using reference point: " + Arrays.toString(this.maximum));
            }

        }
    }

    private void checkRanges() {
        for(int i = 0; i < this.problem.getNumberOfObjectives(); ++i) {
            if (Math.abs(this.minimum[i] - this.maximum[i]) < 1.0E-10D) {
                throw new IllegalArgumentException("objective " + i + " with range = " + Math.abs(this.minimum[i] - this.maximum[i]));
            }
        }
    }

    public NondominatedPopulation normalize(NondominatedPopulation population) {
        NondominatedPopulation result = new NondominatedPopulation() {
            public boolean add(Solution newSolution) {
                return super.forceAddWithoutCheck(newSolution);
            }
        };
        this.normalize(population, result);
        return result;
    }

    public Population normalize(Population population) {
        Population result = new Population();
        this.normalize(population, result);
        return result;
    }

    private void normalize(Population originalSet, Population normalizedSet) {
        Iterator i$ = originalSet.iterator();

        while(true) {
            Solution solution;

            if (!i$.hasNext()) {
                return;
            }

            solution = (Solution)i$.next();
            Solution clone = solution.copy();

            for(int j = 0; j < this.problem.getNumberOfObjectives(); ++j) {
                clone.setObjective(j, (clone.getObjective(j) - this.minimum[j]) / (this.maximum[j] - this.minimum[j]));
            }

            normalizedSet.add(clone);
        }
    }
}
