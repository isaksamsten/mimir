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

import org.briljantframework.Check;
import org.briljantframework.mimir.evaluation.EvaluationContext;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.MeasureCollection;
import org.briljantframework.mimir.evaluation.partition.Partition;

/**
 * @author Isak Karlsson
 */
public class EnsembleEvaluator<In, Out> implements Evaluator<In, Out> {

  private EnsembleEvaluator() {}

  private static EnsembleEvaluator<?, ?> INSTANCE = new EnsembleEvaluator<>();

  @SuppressWarnings("unchecked")
  public static <In, Out> EnsembleEvaluator<In, Out> getInstance() {
    return (EnsembleEvaluator<In, Out>) INSTANCE;
  }

  @Override
  public void accept(EvaluationContext<In, Out> ctx) {
    Check.state(ctx.getPredictor() instanceof Ensemble, "expect ensemble");

    Partition<In, Out> partition = ctx.getPartition();
    Ensemble<In, Out> predictor = (Ensemble<In, Out>) ctx.getPredictor();
    EnsembleClassifierMeasure em = new EnsembleClassifierMeasure(predictor,
        partition.getTrainingData(), partition.getTrainingTarget(), partition.getValidationData(),
        partition.getValidationTarget());

    MeasureCollection measureCollection = ctx.getMeasureCollection();
    measureCollection.add("ensembleVariance", em.getVariance());
    measureCollection.add("ensembleBias", em.getBias());
    measureCollection.add("ensembleMse", em.getMeanSquareError());
    measureCollection.add("baseModelError", em.getBaseModelError());

    measureCollection.add("ensembleStrength", em.getStrength());
    measureCollection.add("ensembleCorrelation", em.getCorrelation());
    measureCollection.add("ensembleQuality", em.getQuality());
    measureCollection.add("ensembleErrorBound", em.getErrorBound());

    measureCollection.add("oobError", em.getOobError());
  }

  @Override
  public String toString() {
    return "EnsembleEvaluator";
  }
}
