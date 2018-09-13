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

import java.util.List;

/**
 * @author Isak Karlsson
 */
public class ClassifierEvaluator<In, Out> implements Evaluator<In, Out> {

  private ClassifierEvaluator() {}

  private static ClassifierEvaluator<?, ?> evaluator = new ClassifierEvaluator<>();

  @SuppressWarnings("unchecked")
  public static <In, Out> ClassifierEvaluator<In, Out> getInstance() {
    return (ClassifierEvaluator<In, Out>) evaluator;
  }

  @Override
  public void accept(EvaluationContext<In, Out> ctx) {
    Check.state(ctx.getPredictor() instanceof Classifier, "expect classifier");
    List<?> predictions = ctx.getPredictions();
    List<?> truth = ctx.getPartition().getValidationTarget();
    Classifier<In, Out> classifier = (Classifier<In, Out>) ctx.getPredictor();
    ClassifierMeasure cm =
        new ClassifierMeasure(classifier.getClasses(), predictions, truth, ctx.getEstimates());

    ctx.getMeasureCollection().add("accuracy", cm.getAccuracy());
    ctx.getMeasureCollection().add("error", cm.getError());
    ctx.getMeasureCollection().add("precision", cm.getPrecision());
    ctx.getMeasureCollection().add("recall", cm.getRecall());
    ctx.getMeasureCollection().add("f1Measure", cm.getF1Measure());
    ctx.getMeasureCollection().add("aucRoc", cm.getAreaUnderRocCurve());
    ctx.getMeasureCollection().add("brierScore", cm.getBrierScore());
  }

  @Override
  public String toString() {
    return "ClassifierEvaluator";
  }
}
