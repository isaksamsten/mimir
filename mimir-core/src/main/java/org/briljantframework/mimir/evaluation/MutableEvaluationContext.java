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
package org.briljantframework.mimir.evaluation;

import java.util.Objects;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.evaluation.partition.Partition;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class MutableEvaluationContext<In, Out> implements EvaluationContext<In, Out> {

  private final ImmutableEvaluationContext evaluationContext = new ImmutableEvaluationContext();

  private Output<Out> predictions;
  private Predictor<In, Out> predictor;
  private Partition<In, Out> partition;
  private DoubleArray estimates;

  public MutableEvaluationContext() {}

  public void setPartition(Partition<In, Out> partition) {
    this.partition = Objects.requireNonNull(partition, "requires a partition");
  }

  /**
   * Set the out of sample predictions made by the predictor
   * 
   * @param predictions the out of sample predictions
   */
  public void setPredictions(Output<Out> predictions) {
    this.predictions = Objects.requireNonNull(predictions, "requires predictions");
  }

  /**
   * Set the out-of-sample probability estimates
   * 
   * @param estimates the probability estimates
   */
  public void setEstimates(DoubleArray estimates) {
    this.estimates = Objects.requireNonNull(estimates, "requires an estimation");
  }

  /**
   * Set the predictor
   * 
   * @param predictor the predictor
   */
  public void setPredictor(Predictor<In, Out> predictor) {
    this.predictor = Objects.requireNonNull(predictor, "requires a predictor");
  }

  @Override
  public Partition<In, Out> getPartition() {
    return partition;
  }

  @Override
  public Output<Out> getPredictions() {
    return predictions;
  }

  @Override
  public DoubleArray getEstimates() {
    return estimates;
  }

  public Predictor<In, Out> getPredictor() {
    return predictor;
  }

  @Override
  public MeasureCollection getMeasureCollection() {
    throw new UnsupportedOperationException();
  }

  public EvaluationContext<In, Out> getEvaluationContext() {
    return evaluationContext;
  }

  private class ImmutableEvaluationContext implements EvaluationContext<In, Out> {

    private final MeasureCollection measureCollection = new MeasureCollection();

    @Override
    public Partition<In, Out> getPartition() {
      return partition;
    }

    @Override
    public Output<Out> getPredictions() {
      return predictions;
    }

    @Override
    public DoubleArray getEstimates() {
      return estimates;
    }

    @Override
    public Predictor<In, Out> getPredictor() {
      return predictor;
    }

    @Override
    public MeasureCollection getMeasureCollection() {
      return measureCollection;
    }
  }
}
