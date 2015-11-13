package org.mimirframework.classification.conformal.evaluation;

import java.util.List;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.conformal.ConformalClassifier;
import org.mimirframework.evaluation.EvaluationContext;
import org.mimirframework.evaluation.MutableEvaluationContext;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.evaluation.partition.FoldPartitioner;
import org.mimirframework.evaluation.partition.Partition;
import org.mimirframework.evaluation.partition.Partitioner;
import org.mimirframework.evaluation.partition.SplitPartitioner;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class ConformalClassifierValidator<P extends ConformalClassifier>
    extends Validator<P> {

  private final double calibrationSize;
  private final List<Double> confidences;

  public ConformalClassifierValidator(Partitioner partitioner, double calibrationSize,
                                      List<Double> confidences) {
    super(partitioner);
    this.calibrationSize = calibrationSize;
    this.confidences = confidences;
  }

  public ConformalClassifierValidator(Partitioner partitioner, double calibrationSize) {
    super(partitioner);
    this.calibrationSize = calibrationSize;
    this.confidences = Arrays.linspace(0.01, 0.1, 9).toList();
  }

  @Override
  protected void evaluate(EvaluationContext<P> evaluationContext, int fold) {
    for (Double confidence : confidences) {
      evaluationContext.getMeasureCollection().add("significance", confidence);
      evaluationContext.getMeasureCollection().add("fold", fold);
      new ConformalClassifierEvaluator(confidence).accept(evaluationContext);
      acceptEvaluators(evaluationContext);
    }
  }

  @Override
  protected P fit(Predictor.Learner<? extends P> learner, DataFrame x, Vector y) {
    SplitPartitioner partitioner = new SplitPartitioner(calibrationSize);
    Partition partition = partitioner.partition(x, y).iterator().next();
    P fit = learner.fit(partition.getTrainingData(), partition.getTrainingTarget());
    fit.calibrate(partition.getValidationData(), partition.getValidationTarget());
    return fit;
  }

  @Override
  protected void predict(MutableEvaluationContext<? extends P> ctx) {
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
    return new ConformalClassifierValidator<T>(new FoldPartitioner(folds), calibrationSize,
        significance.toList());
  }
}
