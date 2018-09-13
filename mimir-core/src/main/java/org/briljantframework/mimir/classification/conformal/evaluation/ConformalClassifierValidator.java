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

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.conformal.ConformalClassifier;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.evaluation.EvaluationContext;
import org.briljantframework.mimir.evaluation.MutableEvaluationContext;
import org.briljantframework.mimir.evaluation.partition.FoldPartitioner;
import org.briljantframework.mimir.evaluation.partition.Partitioner;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public abstract class ConformalClassifierValidator<In, Out, P extends ConformalClassifier<In, Out>> {
  private final List<? extends ConformalClassifierEvaluator<In, Out>> conformalEvaluators;

  protected ConformalClassifierValidator(Partitioner<In, Out> partitioner,
      DoubleArray confidences) {
    // super(partitioner);
    this.conformalEvaluators = confidences.doubleStream()
        .mapToObj(i -> new ConformalClassifierEvaluator<In, Out>(i)).collect(Collectors.toList());
  }

  protected ConformalClassifierValidator(Partitioner<In, Out> partitioner) {
    this(partitioner, DoubleArray.range(0.01, 0.11, 0.01));
  }

  // @Override
  protected void evaluate(EvaluationContext<In, Out> evaluationContext, int fold) {
    for (ConformalClassifierEvaluator<In, Out> evaluator : conformalEvaluators) {
      evaluationContext.getMeasureCollection().add("fold", fold);
      evaluator.accept(evaluationContext);
      // acceptEvaluators(evaluationContext);
    }
  }

  // @Override
  protected void predict(MutableEvaluationContext<In, Out> ctx) {
    Check.state(ctx.getPredictor() instanceof ConformalClassifier);
    Input<? extends In> validationData = ctx.getPartition().getValidationData();
    List<Out> predictions = new ArrayList<>();
    for (int i = 0; i < validationData.size(); i++) {
      predictions.add(null);
    }
    ctx.setPredictions(predictions);
    @SuppressWarnings("unchecked")
    ConformalClassifier<In, Out> predictor = (ConformalClassifier<In, Out>) ctx.getPredictor();
    ctx.setEstimates(predictor.estimate(validationData));
  }

  /**
   * Validator for bootstrap enabled conformal classifiers
   *
   * @param folds number of validation folds
   * @param significances the significance levels to evaluate
   * @return a validator
   */
  public static <In, Out> BootstrapConformalClassifierValidator<In, Out> crossValidator(int folds,
      DoubleArray significances) {
    return new BootstrapConformalClassifierValidator<In, Out>(new FoldPartitioner<>(folds),
        significances);
  }

  /**
   * Validator for bootstrap enabled conformal classifiers (for the significance levels [0.01, 0.1])
   *
   * @param folds number of validation folds
   * @return a validator
   */
  public static <In, Out> BootstrapConformalClassifierValidator<In, Out> crossValidator(int folds) {
    return crossValidator(folds, DoubleArray.range(0.01, 0.11, 0.01));
  }

  /**
   * Returns a k-fold cross validator for evaluating conformal classifiers. For each fold, the
   * specified calibration set size is used.
   *
   * @param folds the number of folds
   * @param calibrationSize the calibration set size (in each fold)
   * @param significance the confidence for which to acceptEvaluators the
   * @return a new validator for evaluating conformal classifiers of the specified type
   */
  public static <In, Out> InductiveConformalClassifierValidator<In, Out> crossValidator(int folds,
      double calibrationSize, DoubleArray significance) {
    return new InductiveConformalClassifierValidator<In, Out>(new FoldPartitioner<>(folds),
        significance, calibrationSize);
  }

  /**
   * Returns a k-fold cross validator for evaluating inductive conformal classifiers.
   *
   * @param folds the number of folds
   * @param calibrationSize the calibration set size
   * @return a new validator
   */
  public static <In, Out> InductiveConformalClassifierValidator<In, Out> crossValidator(int folds,
      double calibrationSize) {
    return crossValidator(folds, calibrationSize, DoubleArray.range(0.01, 0.11, 0.01));
  }
}
