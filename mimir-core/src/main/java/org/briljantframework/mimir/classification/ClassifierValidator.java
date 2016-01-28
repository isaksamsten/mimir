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

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.evaluation.Evaluator;
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
public class ClassifierValidator<T extends Classifier> extends Validator<T> {

  /**
   * The default evaluators for classifiers
   */
  private static final Set<? extends Evaluator<? super Classifier>> EVALUATORS =
      new HashSet<>(Collections.singletonList(ClassifierEvaluator.INSTANCE));

  public ClassifierValidator(Set<? extends Evaluator<? super T>> evaluators,
      Partitioner partitioner) {
    super(evaluators, partitioner);
  }

  public ClassifierValidator(Partitioner partitioner) {
    super(partitioner);
  }

  @Override
  protected T fit(Predictor.Learner<? extends T> learner, DataFrame x, Vector y) {
    return learner.fit(x, y);
  }

  @Override
  protected void predict(MutableEvaluationContext<? extends T> ctx) {
    T p = ctx.getPredictor();
    Partition partition = ctx.getEvaluationContext().getPartition();
    DataFrame x = partition.getValidationData();
    Vector y = partition.getValidationTarget();
    Vector.Builder builder = y.newBuilder();

    // For the case where the classifier reports the ESTIMATOR characteristic
    // improve the performance by avoiding to recompute the classifications twice.
    if (p.getCharacteristics().contains(ClassifierCharacteristic.ESTIMATOR)) {
      Vector classes = p.getClasses();
      DoubleArray estimate = p.estimate(x);
      ctx.setEstimates(estimate);
      for (int i = 0; i < estimate.rows(); i++) {
        builder.loc().set(i, classes, Arrays.argmax(estimate.getRow(i)));
      }
      ctx.setPredictions(builder.build());
    } else {
      ctx.setPredictions(p.predict(x));
    }
  }

  public static <T extends Classifier> ClassifierValidator<T> holdoutValidator(DataFrame testX,
      Vector testY) {
    return createValidator((x, y) -> Collections.singleton(new Partition(x, testX, y, testY)));
  }

  public static <T extends Classifier> ClassifierValidator<T> splitValidator(double testFraction) {
    return createValidator(new SplitPartitioner(testFraction));
  }

  public static <T extends Classifier> ClassifierValidator<T> leaveOneOutValidator() {
    return createValidator(LOO_PARTITIONER);
  }

  public static <T extends Classifier> ClassifierValidator<T> crossValidator(int folds) {
    return createValidator(new FoldPartitioner(folds));
  }

  private static <T extends Classifier> ClassifierValidator<T> createValidator(
      Partitioner partitioner) {
    return new ClassifierValidator<T>(EVALUATORS, partitioner);
  }
}
