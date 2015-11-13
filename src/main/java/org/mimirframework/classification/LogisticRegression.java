package org.mimirframework.classification;

import static org.mimirframework.classification.optimization.OptimizationUtils.logistic;

import java.util.Collections;
import java.util.Objects;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.briljantframework.optimize.DifferentialFunction;
import org.briljantframework.optimize.LimitedMemoryBfgsOptimizer;
import org.briljantframework.optimize.NonlinearOptimizer;
import org.mimirframework.classification.optimization.BinaryLogisticFunction;
import org.mimirframework.classification.optimization.MultiClassLogisticFunction;
import org.mimirframework.evaluation.EvaluationContext;
import org.mimirframework.evaluation.MeasureSample;
import org.mimirframework.supervised.Characteristic;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class LogisticRegression extends AbstractClassifier {

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
      Vector classes) {
    super(classes);
    this.names = names;
    this.coefficients = coefficients;
    this.logLoss = logLoss;
  }

  @Override
  public DoubleArray estimate(Vector record) {
    DoubleArray x = DoubleArray.zeros(record.size() + 1);
    x.set(0, 1); // set the intercept
    for (int i = 0; i < record.size(); i++) {
      x.set(i + 1, record.loc().getAsDouble(i));
    }

    Vector classes = getClasses();
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
    return Collections
.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  @Override
  public String toString() {
    return "LogisticRegression{" + "coefficients=" + coefficients + ", logLoss=" + logLoss + '}';
  }

  public static final class Configurator
 implements Classifier.Configurator<Learner> {

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

  public static class Evaluator
      implements org.mimirframework.evaluation.Evaluator<LogisticRegression> {

    @Override
    public void accept(EvaluationContext<? extends LogisticRegression> ctx) {
      ctx.getMeasureCollection().add("logLoss",
 MeasureSample.IN_SAMPLE,
          ctx.getPredictor().getLogLoss());

      // TODO: compute log-loss out-sample
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
  public static class Learner implements Predictor.Learner<LogisticRegression> {

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
      this.optimizer = new LimitedMemoryBfgsOptimizer(20, 100, 1E-5);
    }

    // @Override
    // public Evaluator<LogisticRegression> getEvaluator() {
    // return ctx -> ctx.getMeasureCollection().add(Measure.LOG_LOSS,
    // ctx.getPredictor().getLogLoss());
    // }

    @Override
    public String toString() {
      return "LogisticRegression.Learner{" + "regularization=" + regularization + ", optimizer="
          + optimizer + '}';
    }

    @Override
    public LogisticRegression fit(DataFrame df, Vector target) {
      int n = df.rows();
      int m = df.columns();
      Check.argument(n == target.size(),
          "The number of training instances must equal the number of targets");
      Vector classes = Vectors.unique(target);
      DoubleArray x = constructInputMatrix(df, n, m);
      IntArray y = Arrays.newIntArray(target.size());
      for (int i = 0; i < y.size(); i++) {
        y.set(i, Vectors.find(classes, target, i));
      }
      DoubleArray theta;
      DifferentialFunction objective;
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
      df.getColumnIndex().keySet().forEach(names::add);
      return new LogisticRegression(names.build(), theta, logLoss, classes);
    }

    protected DoubleArray constructInputMatrix(DataFrame df, int n, int m) {
      DoubleArray x = DoubleArray.zeros(n, m + 1);
      for (int i = 0; i < n; i++) {
        x.set(i, 0, 1);
        for (int j = 0; j < df.columns(); j++) {
          double v = df.loc().getAsDouble(i, j);
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
