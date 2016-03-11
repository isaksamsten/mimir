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
import java.util.List;
import java.util.Set;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.OutputList;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.MutableEvaluationContext;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.evaluation.partition.*;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierValidator<In, T extends Classifier<In>> extends Validator<In, Object, T> {

  public ClassifierValidator(Set<? extends Evaluator<In, Object, ? super T>> evaluators,
      Partitioner<In, Object> partitioner) {
    super(evaluators, partitioner);
  }

  public ClassifierValidator(Partitioner<In, Object> partitioner) {
    super(partitioner);
  }

  @Override
  protected T fit(Predictor.Learner<In, Object, ? extends T> learner, Input<In> x,
      Output<Object> y) {
    return learner.fit(x, y);
  }

  @Override
  protected void predict(MutableEvaluationContext<In, Object, ? extends T> ctx) {
    T p = ctx.getPredictor();
    Partition<In, Object> partition = ctx.getEvaluationContext().getPartition();
    Input<In> x = partition.getValidationData();
    Output<Object> y = partition.getValidationTarget();
    OutputList<Object> predictions = new OutputList<>();

    // For the case where the classifier reports the ESTIMATOR characteristic
    // improve the performance by avoiding to recompute the classifications twice.
    if (p.getCharacteristics().contains(ClassifierCharacteristic.ESTIMATOR)) {
      List<?> classes = p.getClasses();
      DoubleArray estimate = p.estimate(x);
      ctx.setEstimates(estimate);
      for (int i = 0; i < estimate.rows(); i++) {
        predictions.add(classes.get(Arrays.argmax(estimate.getRow(i))));
      }
      ctx.setPredictions(predictions);
    } else {
      ctx.setPredictions(p.predict(x));
    }
  }

  public static <In, T extends Classifier<In>> ClassifierValidator<In, T> holdoutValidator(
      Input<In> testX, Output<Object> testY) {
    return createValidator((x, y) -> Collections.singleton(new Partition<>(x, testX, y, testY)));
  }

  public static <In, T extends Classifier<In>> ClassifierValidator<In, T> splitValidator(
      double testFraction) {
    return createValidator(new SplitPartitioner<>(testFraction));
  }

  public static <In, T extends Classifier<In>> ClassifierValidator<In, T> leaveOneOutValidator() {
    return createValidator(new LeaveOneOutPartitioner<>());
  }

  public static <In, T extends Classifier<In>> ClassifierValidator<In, T> crossValidator(
      int folds) {
    return createValidator(new FoldPartitioner<>(folds));
  }

  private static <In, T extends Classifier<In>> ClassifierValidator<In, T> createValidator(
      Partitioner<In, Object> partitioner) {
    ClassifierValidator<In, T> v = new ClassifierValidator<>(partitioner);
    v.add(new ClassifierEvaluator<>());
    return v;
  }
}
