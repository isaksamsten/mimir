package org.mimirframework.classification.conformal.evaluation;

import org.apache.commons.math3.util.Precision;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.conformal.ConformalClassifier;
import org.mimirframework.evaluation.EvaluationContext;
import org.mimirframework.evaluation.Evaluator;
import org.mimirframework.evaluation.MeasureCollection;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ConformalClassifierEvaluator implements Evaluator<ConformalClassifier> {

  private final double significance;

  public ConformalClassifierEvaluator(double significance) {
    this.significance = significance;
  }

  @Override
  public void accept(EvaluationContext<? extends ConformalClassifier> ctx) {
    Vector truth = ctx.getPartition().getValidationTarget();
    Vector classes = ctx.getPredictor().getClasses();
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
}
