package org.mimirframework.classification.conformal.evaluation;

import java.util.List;
import java.util.stream.Collectors;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.conformal.BootstrapConformalClassifier;
import org.mimirframework.classification.conformal.ConformalClassifier;
import org.mimirframework.classification.conformal.InductiveConformalClassifier;
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
public abstract class ConformalClassifierValidator<P extends ConformalClassifier> extends
    Validator<P> {
  private final List<ConformalClassifierEvaluator> conformalEvaluators;

  protected ConformalClassifierValidator(Partitioner partitioner, DoubleArray confidences) {
    super(partitioner);
    this.conformalEvaluators =
        confidences.stream().mapToObj(ConformalClassifierEvaluator::new)
            .collect(Collectors.toList());
  }

  protected ConformalClassifierValidator(Partitioner partitioner) {
    this(partitioner, DoubleArray.range(0.01, 0.11, 0.01));
  }

  @Override
  protected void evaluate(EvaluationContext<P> evaluationContext, int fold) {
    for (ConformalClassifierEvaluator evaluator : conformalEvaluators) {
      evaluationContext.getMeasureCollection().add("fold", fold);
      evaluator.accept(evaluationContext);
      acceptEvaluators(evaluationContext);
    }
  }

  @Override
  protected void predict(MutableEvaluationContext<? extends P> ctx) {
    DataFrame validationData = ctx.getPartition().getValidationData();
    ctx.setPredictions(Vector.singleton(null, validationData.rows()));
    ctx.setEstimates(ctx.getPredictor().estimate(validationData));
  }

  /**
   * Validator for bootstrap enabled conformal classifiers
   *
   * @param folds number of validation folds
   * @param significances the significance levels to evaluate
   * @param <T> the type of bootstrap conformal classifier
   * @return a validator
   */
  public static <T extends BootstrapConformalClassifier> ConformalClassifierValidator<T> crossValidator(
      int folds, DoubleArray significances) {
    return validator(new FoldPartitioner(folds), significances);
  }


  /**
   * Validator for bootstrap enabled conformal classifiers (for the significance levels [0.01, 0.1])
   *
   * @param folds number of validation folds
   * @param <T> the type of bootstrap conformal classifier
   * @return a validator
   */
  public static <T extends BootstrapConformalClassifier> ConformalClassifierValidator<T> crossValidator(
      int folds) {
    return crossValidator(folds, DoubleArray.range(0.01, 0.11, 0.01));
  }

  /**
   * Validator for bootstrap enabled conformal classifiers with the traning data partitioned
   * according to the given partitioner.
   * 
   * @param partitioner the partitioner
   * @param significances the significance levels
   * @param <T> the type
   * @return a validator
   */
  public static <T extends BootstrapConformalClassifier> ConformalClassifierValidator<T> validator(
      Partitioner partitioner, DoubleArray significances) {
    return new ConformalClassifierValidator<T>(partitioner, significances) {
      @Override
      protected T fit(Predictor.Learner<? extends T> learner, DataFrame x, Vector y) {
        return learner.fit(x, y);
      }
    };
  }

  /**
   * Validator for inductive conformal classifiers with the training data partitioned according to
   * the given partitioner.
   * 
   * @param partitioner the partitioner
   * @param calibrationSize the calibration set size (as a fraction of the availiable training data
   *        in each fold)
   * @param significance the significance levels
   * @param <T> the type of conformal classifier
   * @return a validator
   */
  public static <T extends InductiveConformalClassifier> ConformalClassifierValidator<T> validator(
      Partitioner partitioner, double calibrationSize, DoubleArray significance) {
    return new ConformalClassifierValidator<T>(partitioner, significance) {

      @Override
      protected T fit(Predictor.Learner<? extends T> learner, DataFrame x, Vector y) {
        SplitPartitioner partitioner = new SplitPartitioner(calibrationSize);
        Partition partition = partitioner.partition(x, y).iterator().next();
        T icc = learner.fit(partition.getTrainingData(), partition.getTrainingTarget());
        icc.calibrate(partition.getValidationData(), partition.getValidationTarget());
        return icc;
      }
    };
  }

  /**
   * Validator for inductive conformal classifiers with the training data partitioned according to
   * the given partitioner with the default significance levels ([0.01, 0.1])
   * 
   * @param partitioner the partitioner
   * @param calibrationSize the calibration set size
   * @param <T> the type of conformal classifier
   * @return a validator
   */
  public static <T extends InductiveConformalClassifier> ConformalClassifierValidator<T> validator(
      Partitioner partitioner, double calibrationSize) {
    return validator(partitioner, calibrationSize, DoubleArray.range(0.01, 0.11, 0.01));
  }

  /**
   * Returns a k-fold cross validator for evaluating conformal classifiers. For each fold, the
   * specified calibration set size is used.
   *
   * @param folds the number of folds
   * @param calibrationSize the calibration set size (in each fold)
   * @param significance the confidence for which to acceptEvaluators the
   * @param <T> the type of validator
   * @return a new validator for evaluating conformal classifiers of the specified type
   */
  public static <T extends InductiveConformalClassifier> ConformalClassifierValidator<T> crossValidator(
      int folds, double calibrationSize, DoubleArray significance) {
    return validator(new FoldPartitioner(folds), calibrationSize, significance);
  }

  /**
   * Returns a k-fold cross validator for evaluating inductive conformal classifiers.
   * 
   * @param folds the number of folds
   * @param calibrationSize the calibration set size
   * @param <T> the type of inductive conformal classifier
   * @return a new validator
   */
  public static <T extends InductiveConformalClassifier> ConformalClassifierValidator<T> crossValidator(
      int folds, double calibrationSize) {
    return crossValidator(folds, calibrationSize, DoubleArray.range(0.01, 0.11, 0.01));
  }
}
