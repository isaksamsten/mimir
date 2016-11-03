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

import java.util.Collection;
import java.util.Collections;
import java.util.Set;

import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.MutableEvaluationContext;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.evaluation.partition.*;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierValidator<In, Out, T extends Classifier<In, Out>>
    extends Validator<In, Out, T> {

  public ClassifierValidator(Set<? extends Evaluator<In, Out, ? super T>> evaluators,
      Partitioner<In, Out> partitioner) {
    super(evaluators, partitioner);
  }

  public ClassifierValidator(Partitioner<In, Out> partitioner) {
    super(partitioner);
  }

  @Override
  protected T fit(Predictor.Learner<? super In, ? super Out, ? extends T> learner, Input<In> x,
      Output<Out> y) {
    return learner.fit(x, y);
  }

  @Override
  protected void predict(MutableEvaluationContext<In, Out, ? extends T> ctx) {
    T p = ctx.getPredictor();
    Partition<In, Out> partition = ctx.getEvaluationContext().getPartition();
    Input<In> x = partition.getValidationData();
    ArrayOutput<Out> predictions = new ArrayOutput<>();

    // For the case where the classifier reports the ESTIMATOR characteristic
    // improve the performance by avoiding to recompute the classifications twice.
    if (p instanceof ProbabilityEstimator) {
      Array<Out> classes = p.getClasses();
      @SuppressWarnings("unchecked")
      DoubleArray estimate = ((ProbabilityEstimator<In, Out>) p).estimate(x);
      ctx.setEstimates(estimate);
      for (int i = 0; i < estimate.rows(); i++) {
        predictions.add(classes.get(Arrays.argmax(estimate.getRow(i))));
      }
      ctx.setPredictions(predictions);
    } else {
      ctx.setPredictions(p.predict(x));
    }
  }

  public static <In, Out, T extends Classifier<In, Out>> ClassifierValidator<In, Out, T> holdoutValidator(
      Input<? extends In> testX, Output<? extends Out> testY) {
    return createValidator(new Partitioner<In, Out>() {
      @Override
      public Collection<Partition<In, Out>> partition(Input<? extends In> x,
          Output<? extends Out> y) {
        return Collections
            .singleton(new Partition<>(Inputs.unmodifiableInput(x), Inputs.unmodifiableInput(testX),
                Outputs.unmodifiableOutput(y), Outputs.unmodifiableOutput(testY)));
      }
    });
  }

  public static <In, Out, T extends Classifier<In, Out>> ClassifierValidator<In, Out, T> splitValidator(
      double testFraction) {
    return createValidator(new SplitPartitioner<>(testFraction));
  }

  public static <In, Out, T extends Classifier<In, Out>> ClassifierValidator<In, Out, T> leaveOneOutValidator() {
    return createValidator(new LeaveOneOutPartitioner<>());
  }

  public static <In, Out, T extends Classifier<In, Out>> ClassifierValidator<In, Out, T> crossValidator(
      int folds) {
    return createValidator(new FoldPartitioner<>(folds));
  }

  private static <In, Out, T extends Classifier<In, Out>> ClassifierValidator<In, Out, T> createValidator(
      Partitioner<In, Out> partitioner) {
    ClassifierValidator<In, Out, T> v = new ClassifierValidator<>(partitioner);
    v.add(ClassifierEvaluator.getInstance());
    return v;
  }
}
