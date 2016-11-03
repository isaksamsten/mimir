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

import org.apache.commons.math3.util.Precision;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.conformal.ConformalClassifier;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.evaluation.EvaluationContext;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.MeasureCollection;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ConformalClassifierEvaluator<In, Out>
    implements Evaluator<In, Out, ConformalClassifier<In, Out>> {

  private final double significance;

  public ConformalClassifierEvaluator(double significance) {
    this.significance = significance;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    ConformalClassifierEvaluator that = (ConformalClassifierEvaluator) o;
    return Precision.equals(significance, that.significance);
  }

  @Override
  public int hashCode() {
    return Double.hashCode(significance);
  }

  @Override
  public void accept(
      EvaluationContext<? extends In, ? extends Out, ? extends ConformalClassifier<In, Out>> ctx) {
    Output<?> truth = ctx.getPartition().getValidationTarget();
    Array<?> classes = ctx.getPredictor().getClasses();
    DoubleArray scores = ctx.getEstimates();
    ConformalClassifierMeasure cm =
        new ConformalClassifierMeasure(truth, scores, significance, classes);

    MeasureCollection measureCollection = ctx.getMeasureCollection();
    measureCollection.add("significance", significance);
    measureCollection.add("accuracy", cm.getAccuracy());
    measureCollection.add("error", cm.getError());
    measureCollection.add("confidence", cm.getConfidence());
    measureCollection.add("credibility", cm.getCredibility());
    measureCollection.add("singletons", cm.getSingletons());
    measureCollection.add("meanPvalue", cm.getAveragePvalue());
    measureCollection.add("noClasses", cm.getNoClasses());
  }
}
