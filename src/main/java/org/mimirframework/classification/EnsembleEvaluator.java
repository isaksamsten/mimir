package org.mimirframework.classification;

import org.mimirframework.evaluation.EvaluationContext;
import org.mimirframework.evaluation.Evaluator;
import org.mimirframework.evaluation.MeasureCollection;
import org.mimirframework.evaluation.partition.Partition;

/**
 * @author Isak Karlsson
 */
public enum EnsembleEvaluator implements Evaluator<Ensemble> {

  INSTANCE;

  @Override
  public void accept(EvaluationContext<? extends Ensemble> ctx) {
    Partition partition = ctx.getPartition();
    EnsembleClassifierMeasure em = new EnsembleClassifierMeasure(ctx.getPredictor(),
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

    measureCollection.add("oobError", em.getOobErro());
  }

  @Override
  public String toString() {
    return "EnsembleEvaluator";
  }
}
