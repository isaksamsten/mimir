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
package org.briljantframework.mimir.classification.conformal.evaluation;

import java.util.List;
import java.util.function.DoubleFunction;
import java.util.stream.Collectors;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.ArrayOutput;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.classification.conformal.BootstrapConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ConformalClassifier;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.evaluation.EvaluationContext;
import org.briljantframework.mimir.evaluation.MutableEvaluationContext;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.evaluation.partition.FoldPartitioner;
import org.briljantframework.mimir.evaluation.partition.Partition;
import org.briljantframework.mimir.evaluation.partition.Partitioner;
import org.briljantframework.mimir.evaluation.partition.SplitPartitioner;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public abstract class ConformalClassifierValidator<In, P extends ConformalClassifier<In>>
    extends Validator<In, Object, P> {
  private final List<? extends ConformalClassifierEvaluator<In>> conformalEvaluators;

  protected ConformalClassifierValidator(Partitioner<In, Object> partitioner,
      DoubleArray confidences) {
    super(partitioner);
    this.conformalEvaluators = confidences.stream()
        .mapToObj(
            (DoubleFunction<ConformalClassifierEvaluator<In>>) ConformalClassifierEvaluator::new)
        .collect(Collectors.toList());
  }

  protected ConformalClassifierValidator(Partitioner<In, Object> partitioner) {
    this(partitioner, DoubleArray.range(0.01, 0.11, 0.01));
  }

  @Override
  protected void evaluate(EvaluationContext<In, Object, P> evaluationContext, int fold) {
    for (ConformalClassifierEvaluator<In> evaluator : conformalEvaluators) {
      evaluationContext.getMeasureCollection().add("fold", fold);
      evaluator.accept(evaluationContext);
      acceptEvaluators(evaluationContext);
    }
  }

  @Override
  protected void predict(MutableEvaluationContext<In, Object, ? extends P> ctx) {
    Input<? extends In> validationData = ctx.getPartition().getValidationData();
    // ctx.setPredictions(Vector.singleton(null, validationData.size()));
    Output<Object> predictions = new ArrayOutput<>();
    for (int i = 0; i < validationData.size(); i++) {
      predictions.add(null);
    }
    ctx.setPredictions(predictions);
    ctx.setEstimates(ctx.getPredictor().estimate(validationData));
  }

  /**
   * Validator for bootstrap enabled conformal classifiers
   *
   * @param folds number of validation folds
   * @param significances the significance levels to evaluate
   * @param <T> the type of bootstrap conformal classifier
   * @return a validator
   */
  public static <In, T extends BootstrapConformalClassifier<In>> ConformalClassifierValidator<In, T> crossValidator(
      int folds, DoubleArray significances) {
    return validator(new FoldPartitioner<>(folds), significances);
  }


  /**
   * Validator for bootstrap enabled conformal classifiers (for the significance levels [0.01, 0.1])
   *
   * @param folds number of validation folds
   * @param <T> the type of bootstrap conformal classifier
   * @return a validator
   */
  public static <In, T extends BootstrapConformalClassifier<In>> ConformalClassifierValidator<In, T> crossValidator(
      int folds) {
    return crossValidator(folds, DoubleArray.range(0.01, 0.11, 0.01));
  }

  /**
   * Validator for bootstrap enabled conformal classifiers with the traning data partitioned
   * according to the given partitioner.
   * 
   * @param partitioner the partitioner
   * @param significances the significance levels
   * @param <T> the type
   * @return a validator
   */
  public static <In, T extends BootstrapConformalClassifier<In>> ConformalClassifierValidator<In, T> validator(
      Partitioner<In, Object> partitioner, DoubleArray significances) {
    return new ConformalClassifierValidator<In, T>(partitioner, significances) {
      @Override
      protected T fit(Predictor.Learner<In, Object, ? extends T> learner, Input<In> x,
          Output<Object> y) {
        return learner.fit(x, y);
      }
    };
  }

  /**
   * Validator for inductive conformal classifiers with the training data partitioned according to
   * the given partitioner.
   * 
   * @param partitioner the partitioner
   * @param calibrationSize the calibration set size (as a fraction of the availiable training data
   *        in each fold)
   * @param significance the significance levels
   * @param <T> the type of conformal classifier
   * @return a validator
   */
  public static <In, T extends InductiveConformalClassifier<In>> ConformalClassifierValidator<In, T> validator(
      Partitioner<In, Object> partitioner, double calibrationSize, DoubleArray significance) {
    return new ConformalClassifierValidator<In, T>(partitioner, significance) {

      @Override
      protected T fit(Predictor.Learner<In, Object, ? extends T> learner, Input<In> x,
          Output<Object> y) {
        SplitPartitioner<In, Object> partitioner = new SplitPartitioner<>(calibrationSize);
        Partition<In, Object> partition = partitioner.partition(x, y).iterator().next();
        T icc = learner.fit(partition.getTrainingData(), partition.getTrainingTarget());
        icc.calibrate(partition.getValidationData(), partition.getValidationTarget());
        return icc;
      }
    };
  }

  /**
   * Validator for inductive conformal classifiers with the training data partitioned according to
   * the given partitioner with the default significance levels ([0.01, 0.1])
   * 
   * @param partitioner the partitioner
   * @param calibrationSize the calibration set size
   * @param <T> the type of conformal classifier
   * @return a validator
   */
  public static <In, T extends InductiveConformalClassifier<In>> ConformalClassifierValidator<In, T> validator(
      Partitioner<In, Object> partitioner, double calibrationSize) {
    return validator(partitioner, calibrationSize, DoubleArray.range(0.01, 0.11, 0.01));
  }

  /**
   * Returns a k-fold cross validator for evaluating conformal classifiers. For each fold, the
   * specified calibration set size is used.
   *
   * @param folds the number of folds
   * @param calibrationSize the calibration set size (in each fold)
   * @param significance the confidence for which to acceptEvaluators the
   * @param <T> the type of validator
   * @return a new validator for evaluating conformal classifiers of the specified type
   */
  public static <In, T extends InductiveConformalClassifier<In>> ConformalClassifierValidator<In, T> crossValidator(
      int folds, double calibrationSize, DoubleArray significance) {
    return validator(new FoldPartitioner<>(folds), calibrationSize, significance);
  }

  /**
   * Returns a k-fold cross validator for evaluating inductive conformal classifiers.
   * 
   * @param folds the number of folds
   * @param calibrationSize the calibration set size
   * @param <T> the type of inductive conformal classifier
   * @return a new validator
   */
  public static <In, T extends InductiveConformalClassifier<In>> ConformalClassifierValidator<In, T> crossValidator(
      int folds, double calibrationSize) {
    return crossValidator(folds, calibrationSize, DoubleArray.range(0.01, 0.11, 0.01));
  }
}
