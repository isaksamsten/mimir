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
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.evaluation.partition.Partition;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class MutableEvaluationContext<P extends Predictor> implements EvaluationContext<P> {

  private final ImmutableEvaluationContext evaluationContext = new ImmutableEvaluationContext();

  private Vector predictions;
  private P predictor;
  private Partition partition;
  private DoubleArray estimates;

  public MutableEvaluationContext() {}

  public void setPartition(Partition partition) {
    this.partition = Objects.requireNonNull(partition, "requires a partition");
  }

  /**
   * Set the out of sample predictions made by the predictor
   * 
   * @param predictions the out of sample predictions
   */
  public void setPredictions(Vector predictions) {
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
  public void setPredictor(P predictor) {
    this.predictor = Objects.requireNonNull(predictor, "requires a predictor");
  }

  @Override
  public Partition getPartition() {
    return partition;
  }

  @Override
  public Vector getPredictions() {
    return predictions;
  }

  @Override
  public DoubleArray getEstimates() {
    return estimates;
  }

  public P getPredictor() {
    return predictor;
  }

  @Override
  public MeasureCollection getMeasureCollection() {
    throw new UnsupportedOperationException();
  }

  public EvaluationContext<P> getEvaluationContext() {
    return evaluationContext;
  }

  private class ImmutableEvaluationContext implements EvaluationContext<P> {

    private final MeasureCollection measureCollection = new MeasureCollection();

    @Override
    public Partition getPartition() {
      return partition;
    }

    @Override
    public Vector getPredictions() {
      return predictions;
    }

    @Override
    public DoubleArray getEstimates() {
      return estimates;
    }

    // @Override
    // public <T extends Measure, C extends Measure.Builder<T>> C getOrDefault(Class<T> measure,
    // Supplier<C> supplier) {
    // C builder = get(measure);
    // if (builder == null) {
    // builder = supplier.get();
    // builders.put(measure, builder);
    // }
    //
    // return builder;
    // }

    @Override
    public P getPredictor() {
      return predictor;
    }

    // @Override
    // public List<Measure> getMeasures() {
    // List<Measure> measures = new ArrayList<>();
    // getMeasureBuilders().forEachDouble(v -> measures.add(v.build()));
    // return measures;
    // }

    @Override
    public MeasureCollection getMeasureCollection() {
      return measureCollection;
    }
  }
}
