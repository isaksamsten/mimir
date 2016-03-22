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

import java.util.*;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.optimization.BinaryLogisticFunction;
import org.briljantframework.mimir.classification.optimization.MultiClassLogisticFunction;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.evaluation.EvaluationContext;
import org.briljantframework.mimir.evaluation.MeasureSample;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;
import org.briljantframework.optimize.DifferentialMultivariateFunction;
import org.briljantframework.optimize.LimitedMemoryBfgsOptimizer;
import org.briljantframework.optimize.NonlinearOptimizer;

/**
 * @author Isak Karlsson
 */
public class LogisticRegression extends AbstractClassifier<Instance> {

  public enum Measure {
    LOG_LOSS
  }

  private final Vector names;

  /**
   * If {@code getClasses().size()} is larger than {@code 2}, coefficients is a a 2d-array where
   * each column is the coefficients for the the j:th class and the i:th feature.
   *
   * On the other hand, if {@code getClasses().size() <= 2}, coefficients is a 1d-array where each
   * element is the coefficient for the i:th feature.
   */
  private final DoubleArray coefficients;
  private final double logLoss;

  private LogisticRegression(Vector names, DoubleArray coefficients, double logLoss,
      List<?> classes) {
    super(classes);
    this.names = names;
    this.coefficients = coefficients;
    this.logLoss = logLoss;
  }

  @Override
  public DoubleArray estimate(Instance record) {
    DoubleArray x = DoubleArray.zeros(record.size() + 1);
    x.set(0, 1); // set the intercept
    for (int i = 0; i < record.size(); i++) {
      x.set(i + 1, record.getAsDouble(i));
    }

    List<?> classes = getClasses();
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
      probs.divAssign(z);
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
    int i = names.loc().indexOf(coefficient);
    if (i < 0) {
      throw new IllegalArgumentException("Label not found");
    }
    int k = getClasses().size();
    if (k > 2) {
      return Arrays.mean(Arrays.exp(coefficients.getRow(i)));
    } else {
      return Math.exp(coefficients.get(i));
    }

  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  @Override
  public String toString() {
    return "LogisticRegression{" + "coefficients=" + coefficients + ", logLoss=" + logLoss + '}';
  }

  public static final class Configurator
      implements Classifier.Configurator<Instance, Object, Learner> {

    private int iterations = 100;
    private double regularization = 0.01;

    private NonlinearOptimizer optimizer;

    public Configurator() {}

    public Configurator(int iterations) {
      this.iterations = iterations;
    }

    public Configurator setIterations(int it) {
      this.iterations = it;
      return this;
    }

    public Configurator setRegularization(double lambda) {
      this.regularization = lambda;
      return this;
    }

    public void setOptimizer(NonlinearOptimizer optimizer) {
      this.optimizer = optimizer;
    }

    @Override
    public Learner configure() {
      if (optimizer == null) {
        // m ~ 20, [1] pp 252.
        optimizer = new LimitedMemoryBfgsOptimizer(20, iterations, 1E-5);
      }
      return new Learner(this);
    }
  }

  public static class Evaluator implements
      org.briljantframework.mimir.evaluation.Evaluator<Instance, Object, LogisticRegression> {

    @Override
    public void accept(EvaluationContext<? extends Instance, ?, ? extends LogisticRegression> ctx) {
      ctx.getMeasureCollection().add("logLoss", MeasureSample.IN_SAMPLE,
          ctx.getPredictor().getLogLoss());
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
  public static class Learner implements Predictor.Learner<Instance, Object, LogisticRegression> {

    public static final double GRADIENT_TOLERANCE = 1E-5;
    public static final int MAX_ITERATIONS = 100;
    public static final int MEMORY = 20;

    private final double regularization;
    private final NonlinearOptimizer optimizer;

    private Learner(Configurator builder) {
      Check.argument(
          !Double.isNaN(builder.regularization) && !Double.isInfinite(builder.regularization));
      this.regularization = builder.regularization;
      this.optimizer = Objects.requireNonNull(builder.optimizer);
    }

    public Learner() {
      this.regularization = 0.01;
      this.optimizer = new LimitedMemoryBfgsOptimizer(MEMORY, MAX_ITERATIONS, GRADIENT_TOLERANCE);
    }

    public Learner(double regularization) {
      this.regularization = regularization;
      this.optimizer = new LimitedMemoryBfgsOptimizer(MEMORY, MAX_ITERATIONS, GRADIENT_TOLERANCE);
    }

    @Override
    public String toString() {
      return "LogisticRegression.Learner{" + "regularization=" + regularization + ", optimizer="
          + optimizer + '}';
    }

    @Override
    public LogisticRegression fit(Input<? extends Instance> in, Output<?> out) {
      PropertyPreconditions.checkProperties(getRequiredInputProperties(), in);
      Check.argument(Dataset.isAllNumeric(in), "All features must be numeric.");

      int n = in.size();
      int m = in.getProperty(Dataset.FEATURE_SIZE);
      Check.argument(n == out.size(),
          "The number of training instances must equal the number of targets");
      List<?> classes = Outputs.unique(out);
      DoubleArray x = constructInputMatrix(in, n, m);
      IntArray y = Arrays.intArray(out.size());
      for (int i = 0; i < y.size(); i++) {
        y.set(i, classes.indexOf(y.get(i)));
      }
      DoubleArray theta;
      DifferentialMultivariateFunction objective;
      int k = classes.size();
      if (k == 2) {
        objective = new BinaryLogisticFunction(x, y, regularization);
        theta = DoubleArray.zeros(x.columns());
      } else if (k > 2) {
        objective = new MultiClassLogisticFunction(x, y, regularization, k);
        theta = DoubleArray.zeros(x.columns(), k);
      } else {
        throw new IllegalArgumentException(String.format("Illegal classes. k >= 2 (%d >= 2)", k));
      }
      double logLoss = optimizer.optimize(objective, theta);

      Vector.Builder names = Vector.Builder.of(Object.class).add("(Intercept)");
      if (in.getProperties().contains(Dataset.FEATURE_NAMES)) {
        in.getProperty(Dataset.FEATURE_NAMES).forEach(names::add);
      } else {
        for (int i = 0; i < m; i++) {
          names.add(String.valueOf(i));
        }
      }
      return new LogisticRegression(names.build(), theta, logLoss, classes);
    }

    @Override
    public Collection<Property<?>> getRequiredInputProperties() {
      return java.util.Arrays.asList(Dataset.FEATURE_SIZE, Dataset.FEATURE_TYPES);
    }

    protected DoubleArray constructInputMatrix(Input<? extends Instance> input, int n, int m) {
      DoubleArray x = DoubleArray.zeros(n, m + 1);
      for (int i = 0; i < n; i++) {
        x.set(i, 0, 1);
        Instance record = input.get(i);
        for (int j = 0; j < m; j++) {
          double v = record.getAsDouble(j);
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
