package org.mimirframework.classification.conformal.evaluation;

import java.util.List;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.mimirframework.classification.conformal.ConformalClassifier;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class ConformalClassifierValidator<P extends ConformalClassifier>
    extends org.mimirframework.evaluation.Validator<P> {

  private final double calibrationSize;
  private final List<Double> confidences;

  public ConformalClassifierValidator(org.mimirframework.evaluation.partition.Partitioner partitioner, double calibrationSize,
                                      List<Double> confidences) {
    super(partitioner);
    this.calibrationSize = calibrationSize;
    this.confidences = confidences;
  }

  public ConformalClassifierValidator(org.mimirframework.evaluation.partition.Partitioner partitioner, double calibrationSize) {
    super(partitioner);
    this.calibrationSize = calibrationSize;
    this.confidences = Arrays.linspace(0.01, 0.1, 9).toList();
  }

  @Override
  protected void evaluate(org.mimirframework.evaluation.EvaluationContext<P> evaluationContext, int fold) {
    for (Double confidence : confidences) {
      evaluationContext.getMeasureCollection().add("significance", confidence);
      evaluationContext.getMeasureCollection().add("fold", fold);
      new ConformalClassifierEvaluator(confidence).accept(evaluationContext);
      acceptEvaluators(evaluationContext);
    }
  }

  @Override
  protected P fit(org.mimirframework.supervised.Predictor.Learner<? extends P> learner, DataFrame x, Vector y) {
    org.mimirframework.evaluation.partition.SplitPartitioner partitioner = new org.mimirframework.evaluation.partition.SplitPartitioner(calibrationSize);
    org.mimirframework.evaluation.partition.Partition partition = partitioner.partition(x, y).iterator().next();
    P fit = learner.fit(partition.getTrainingData(), partition.getTrainingTarget());
    fit.calibrate(partition.getValidationData(), partition.getValidationTarget());
    return fit;
  }

  @Override
  protected void predict(org.mimirframework.evaluation.MutableEvaluationContext<? extends P> ctx) {
    DataFrame validationData = ctx.getPartition().getValidationData();
    ctx.setPredictions(ctx.getPredictor().predict(validationData));
    ctx.setEstimates(ctx.getPredictor().estimate(validationData));
  }

  /**
   * Returns a k-fold cross validator for evaluating conformal classifiers. For each fold, the
   * specified calibration set size is used. The default {@linkplain org.mimirframework.evaluation.Evaluator evaluator} is the
   * {@link ConformalClassifierEvaluator} (with the specified confidence level)
   *
   * @param folds the number of folds
   * @param calibrationSize the calibration set size (in each fold)
   * @param significance the confidence for which to acceptEvaluators the
   * @param <T> the type of validator
   * @return a new validator for evaluating conformal classifiers of the specified type
   */
  public static <T extends ConformalClassifier> ConformalClassifierValidator<T> crossValidator(
      int folds, double calibrationSize, double[] significance) {
    return crossValidator(folds, calibrationSize, Arrays.newDoubleVector(significance));
  }

  public static <T extends ConformalClassifier> ConformalClassifierValidator<T> crossValidator(
      int folds, double calibrationSize, DoubleArray significance) {
    return new ConformalClassifierValidator<T>(new org.mimirframework.evaluation.partition.FoldPartitioner(folds), calibrationSize,
        significance.toList());
  }
}
