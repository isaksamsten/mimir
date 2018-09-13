/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.briljantframework.mimir.classification;

import static org.briljantframework.mimir.classification.optimization.OptimizationUtils.logistic;

import java.util.HashSet;
import java.util.List;

import org.briljantframework.Check;
import org.briljantframework.DoubleVector;
import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.Properties;
import org.briljantframework.mimir.Property;
import org.briljantframework.mimir.classification.optimization.BinaryLogisticFunction;
import org.briljantframework.mimir.classification.optimization.MultiClassLogisticFunction;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.evaluation.EvaluationContext;
import org.briljantframework.mimir.evaluation.MeasureSample;
import org.briljantframework.mimir.supervised.Predictor;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;
import org.briljantframework.optimize.DifferentialMultivariateFunction;
import org.briljantframework.optimize.LimitedMemoryBfgsOptimizer;
import org.briljantframework.optimize.NonlinearOptimizer;

/**
 * @author Isak Karlsson
 */
public class LogisticRegression<Out> extends AbstractClassifier<Instance, Out>
    implements ProbabilityEstimator<Instance, Out> {

  /**
   * Maximum number of iterations before convergence.
   */
  public static final Property<Integer> MAX_ITERATIONS =
      Property.of("max_iterations", Integer.class, 100);

  public static final Property<Double> REGULARIZATION =
      Property.of("regularization", Double.class, 1.0);
  /**
   * If {@code getClasses().size()} is larger than {@code 2}, coefficients is a a 2d-array where
   * each column is the coefficients for the the j:th class and the i:th feature.
   *
   * On the other hand, if {@code getClasses().size() <= 2}, coefficients is a 1d-array where each
   * element is the coefficient for the i:th feature.
   */
  private final DoubleArray coefficients;

  private final double logLoss;
  private final MultidimensionalSchema schema;

  private LogisticRegression(Array<Out> classes, MultidimensionalSchema schema,
      DoubleArray coefficients, double logLoss) {
    super(classes);
    this.schema = schema;
    this.coefficients = coefficients;
    this.logLoss = logLoss;
  }

  @Override
  public DoubleArray estimate(Instance record) {
    Check.argument(schema.isValid(record), "illegal input");
    return estimateProbability(record.getNumericalAttributes());
  }

  private DoubleArray estimateProbability(DoubleVector record) {
    DoubleArray x = DoubleArray.zeros(record.size() + 1);
    x.set(0, 1); // set the intercept
    for (int i = 0; i < record.size(); i++) {
      x.set(i + 1, record.getDouble(i));
    }
    Array<?> classes = getClasses();
    int k = classes.size();
    if (k > 2) {
      DoubleArray probs = DoubleArray.zeros(k);
      double max = Double.NEGATIVE_INFINITY;
      for (int i = 0; i < k; i++) {
        double prob = Arrays.inner(x, coefficients.getColumn(i));
        if (prob > max) {
          max = prob;
        }
        probs.set(i, prob);
      }

      double z = 0;
      for (int i = 0; i < k; i++) {
        probs.set(i, Math.exp(probs.get(i) - max));
        z += probs.get(i);
      }
      Arrays.scal(1 / z, probs);
      return probs;
    } else {
      double prob = logistic(Arrays.inner(x, coefficients));
      DoubleArray probs = DoubleArray.zeros(2);
      probs.set(0, 1 - prob);
      probs.set(1, prob);
      return probs;
    }
  }

  public DoubleArray getParameters() {
    return coefficients.copy();
  }

  public double getLogLoss() {
    return logLoss;
  }

  public double getOddsRatio(Object coefficient) {
    // int i = names.values().indexOf(coefficient);
    // if (i < 0) {
    // throw new IllegalArgumentException("Label not found");
    // }
    // int k = getClasses().size();
    // if (k > 2) {
    // return Arrays.mean(Arrays.exp(coefficients.getRow(i)));
    // } else {
    // return Math.exp(coefficients.get(i));
    // }
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return "LogisticRegression{" + "coefficients=" + coefficients + ", logLoss=" + logLoss + '}';
  }

  public static class Evaluator
      implements org.briljantframework.mimir.evaluation.Evaluator<Instance, Object> {

    @Override
    public void accept(EvaluationContext<Instance, Object> ctx) {
      Check.state(ctx.getPredictor() instanceof LogisticRegression, "expect logistic regression");
      LogisticRegression<Object> predictor = (LogisticRegression<Object>) ctx.getPredictor();
      ctx.getMeasureCollection().add("logLoss", MeasureSample.IN_SAMPLE, predictor.getLogLoss());
    }
  }

  /**
   * Logistic regression implemented using a quasi newton method based on the limited memory BFGS.
   *
   * <p>
   * References:
   * <ol>
   * <li>Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.</li>
   *
   * </ol>
   *
   * @author Isak Karlsson
   */
  public static class Learner<Out>
      extends Predictor.Learner<Instance, Out, LogisticRegression<Out>> {

    static final double GRADIENT_TOLERANCE = 1E-5;
    static final int MEMORY = 20;

    public Learner(double regularization) {
      set(REGULARIZATION, regularization);
    }

    public Learner() {
      this(0.01);
    }

    @Override
    public LogisticRegression<Out> fit(Input<Instance> in, List<Out> out) {
      Check.argument(in.getSchema() instanceof MultidimensionalSchema, "illegal input schema");
      Check.argument(in.size() == out.size(),
          "The number of training instances must equal the number of targets");

      MultidimensionalSchema schema = (MultidimensionalSchema) in.getSchema();
      Check.argument(!schema.hasCategoricalAttributes(), "can't handle categorical attributes");
      Array<Out> classes = Array.copyOf(new HashSet<>(out));
      // DoubleArray x = constructInputMatrix(in, in.size(), in.getProperty(Dataset.FEATURE_SIZE));
      DoubleArray x =
          Arrays.hstack(DoubleArray.ones(in.size(), 1), schema.getNumericalAttributes(in));
      IntArray y = Arrays.intArray(out.size());
      for (int i = 0; i < y.size(); i++) {
        y.set(i, classes.indexOf(out.get(i)));
      }
      return train(getParameters(), x, y, classes, schema);
    }

    public LogisticRegression<Integer> fit(DoubleArray x, IntArray y) {
      Check.argument(x.isMatrix(), "Require 2d-array");
      Check.argument(x.rows() == y.size(),
          "The number of training instance must equal the number of targets");
      Array<Integer> classes = Array.copyOf(new HashSet<>(y));
      x = Arrays.hstack(DoubleArray.ones(x.rows(), 1), x);
      return train(getParameters(), x, y, classes, new MultidimensionalSchema(x.columns(), 0));
    }

    private static <Out> LogisticRegression<Out> train(Properties props, DoubleArray x, IntArray y,
        Array<Out> classes, MultidimensionalSchema schema) {
      DoubleArray theta;
      DifferentialMultivariateFunction objective;
      int k = classes.size();
      Double lambda = props.get(REGULARIZATION);
      if (k == 2) {
        objective = new BinaryLogisticFunction(x, y, lambda);
        theta = DoubleArray.zeros(x.columns());
      } else if (k > 2) {
        objective = new MultiClassLogisticFunction(x, y, lambda, k);
        theta = DoubleArray.zeros(x.columns(), k);
      } else {
        throw new IllegalArgumentException(String.format("Illegal classes. k >= 2 (%d >= 2)", k));
      }

      Integer maxIterations = props.getOrDefault(MAX_ITERATIONS);
      NonlinearOptimizer optimizer =
          new LimitedMemoryBfgsOptimizer(MEMORY, maxIterations, GRADIENT_TOLERANCE);
      double logLoss = optimizer.optimize(objective, theta);

      // Series.MultidimensionalBuilder names =
      // Series.MultidimensionalBuilder.of(String.class).add("(Intercept)");
      // int independentVariables = x.columns() - 1;
      // if (featureNames != null && featureNames.size() == independentVariables) {
      // featureNames.forEach(names::add);
      // } else {
      // for (int i = 0; i < independentVariables; i++) {
      // names.add(String.valueOf(i));
      // }
      // }
      return new LogisticRegression<>(classes, schema, theta, logLoss);
    }

    DoubleArray constructInputMatrix(Input<? extends DoubleVector> input, int n, int m) {
      DoubleArray x = DoubleArray.zeros(n, m + 1);
      for (int i = 0; i < n; i++) {
        x.set(i, 0, 1);
        DoubleVector record = input.get(i);
        for (int j = 0; j < m; j++) {
          double v = record.getDouble(j);
          if (Is.NA(v) || Double.isNaN(v)) {
            throw new IllegalArgumentException(
                String.format("Illegal input value at (%d, %d)", i, j - 1));
          }
          x.set(i, j + 1, v);
        }
      }
      return x;
    }
  }
}
