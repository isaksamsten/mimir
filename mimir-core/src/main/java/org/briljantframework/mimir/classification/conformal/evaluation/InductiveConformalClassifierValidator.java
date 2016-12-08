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

import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.evaluation.partition.Partition;
import org.briljantframework.mimir.evaluation.partition.Partitioner;
import org.briljantframework.mimir.evaluation.partition.SplitPartitioner;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * Created by isak on 2016-12-07.
 */
public class InductiveConformalClassifierValidator<In, Out>
    extends ConformalClassifierValidator<In, Out, InductiveConformalClassifier<In, Out>> {

  private final double calibrationSize;

  protected InductiveConformalClassifierValidator(Partitioner<In, Out> partitioner,
      DoubleArray confidences, double calibrationSize) {
    super(partitioner, confidences);
    this.calibrationSize = calibrationSize;
  }

  protected InductiveConformalClassifierValidator(Partitioner<In, Out> partitioner,
      double calibrationSize) {
    super(partitioner);
    this.calibrationSize = calibrationSize;
  }

  @Override
  protected InductiveConformalClassifier<In, Out> fit(
      Predictor.Learner<? super In, ? super Out, ? extends InductiveConformalClassifier<In, Out>> learner,
      Input<In> x, Output<Out> y) {
    SplitPartitioner<In, Out> partitioner = new SplitPartitioner<>(calibrationSize);
    Partition<In, Out> partition = partitioner.partition(x, y).iterator().next();

    InductiveConformalClassifier<In, Out> icc =
        learner.fit(partition.getTrainingData(), partition.getTrainingTarget());
    icc.calibrate(partition.getValidationData(), partition.getValidationTarget());
    return icc;
  }
}
