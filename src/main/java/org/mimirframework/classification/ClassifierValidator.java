package org.mimirframework.classification;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.briljantframework.array.DoubleArray;
import org.mimirframework.classifier.evaluation.ClassifierEvaluator;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.evaluation.Evaluator;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.evaluation.partition.Partition;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierValidator<T extends Classifier> extends Validator<T> {

  /**
   * The default evaluators for classifiers
   */
  private static final Set<? extends Evaluator<? super Classifier>> EVALUATORS =
      new HashSet<>(Collections.singletonList(ClassifierEvaluator.getInstance()));

  public ClassifierValidator(Set<? extends Evaluator<? super T>> evaluators,
      org.mimirframework.evaluation.partition.Partitioner partitioner) {
    super(evaluators, partitioner);
  }

  public ClassifierValidator(org.mimirframework.evaluation.partition.Partitioner partitioner) {
    super(partitioner);
  }

  @Override
  protected T fit(Predictor.Learner<? extends T> learner, DataFrame x, Vector y) {
    return learner.fit(x, y);
  }

  @Override
  protected void predict(org.mimirframework.evaluation.MutableEvaluationContext<? extends T> ctx) {
    T p = ctx.getPredictor();
    Partition partition = ctx.getEvaluationContext().getPartition();
    DataFrame x = partition.getValidationData();
    Vector y = partition.getValidationTarget();
    Vector.Builder builder = y.newBuilder();

    // For the case where the classifier reports the ESTIMATOR characteristic
    // improve the performance by avoiding to recompute the classifications twice.
    if (p.getCharacteristics().contains(ClassifierCharacteristic.ESTIMATOR)) {
      Vector classes = p.getClasses();
      DoubleArray estimate = p.estimate(x);
      ctx.setEstimates(estimate);
      for (int i = 0; i < estimate.rows(); i++) {
        builder.loc().set(i, classes,
            org.briljantframework.array.Arrays.argmax(estimate.getRow(i)));
      }
      ctx.setPredictions(builder.build());
    } else {
      ctx.setPredictions(p.predict(x));
    }
  }

  public static <T extends Classifier> ClassifierValidator<T> holdout(DataFrame testX,
      Vector testY) {
    return createValidator((x, y) -> Collections.singleton(new Partition(x, testX, y, testY)));
  }

  public static <T extends Classifier> ClassifierValidator<T> splitValidation(double testFraction) {
    return createValidator(new org.mimirframework.evaluation.partition.SplitPartitioner(testFraction));
  }

  public static <T extends Classifier> ClassifierValidator<T> leaveOneOutValidation() {
    return createValidator(LOO_PARTITIONER);
  }

  public static <T extends Classifier> ClassifierValidator<T> crossValidation(int folds) {
    return createValidator(new org.mimirframework.evaluation.partition.FoldPartitioner(folds));
  }

  private static <T extends Classifier> ClassifierValidator<T> createValidator(
      org.mimirframework.evaluation.partition.Partitioner partitioner) {
    return new ClassifierValidator<T>(EVALUATORS, partitioner);
  }
}
