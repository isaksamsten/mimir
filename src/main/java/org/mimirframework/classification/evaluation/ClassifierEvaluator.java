/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.classification.evaluation;

import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.ClassifierMeasure;
import org.mimirframework.evaluation.EvaluationContext;
import org.mimirframework.evaluation.Evaluator;

/**
 * @author Isak Karlsson
 */
public enum ClassifierEvaluator implements Evaluator<Classifier> {
  INSTANCE;

  @Override
  public void accept(EvaluationContext<? extends Classifier> ctx) {
    Vector predictions = ctx.getPredictions();
    Vector truth = ctx.getPartition().getValidationTarget();
    ClassifierMeasure cm = new ClassifierMeasure(truth, predictions, ctx.getEstimates(),
        ctx.getPredictor().getClasses());

    ctx.getMeasureCollection().add("accuracy", cm.getAccuracy());
    ctx.getMeasureCollection().add("error", cm.getError());
    ctx.getMeasureCollection().add("precision", cm.getPrecision());
    ctx.getMeasureCollection().add("recall", cm.getRecall());
    ctx.getMeasureCollection().add("aucRoc", cm.getAreaUnderRocCurve());
    ctx.getMeasureCollection().add("brierScore", cm.getBrierScore());
  }

  @Override
  public String toString() {
    return "ClassifierEvaluator";
  }
}
