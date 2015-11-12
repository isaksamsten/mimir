package org.mimirframework.evaluation;

import java.util.Random;

import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.NearestNeighbours;
import org.mimirframework.classifier.evaluation.ClassifierEvaluator;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ValidatorTest {


  @Test
  public void testCreateValidator() throws Exception {
    org.mimirframework.evaluation.partition.Partitioner partitioner = new org.mimirframework.evaluation.partition.FoldPartitioner(10);
    Validator<Classifier> validator = new Validator<Classifier>(partitioner) {
      @Override
      protected org.mimirframework.classification.Classifier fit(org.mimirframework.supervised.Predictor.Learner<? extends Classifier> learner, DataFrame x, Vector y) {
        return learner.fit(x, y);
      }

      @Override
      protected void predict(MutableEvaluationContext<? extends org.mimirframework.classification.Classifier> ctx) {
        DataFrame validationData = ctx.getPartition().getValidationData();
        org.mimirframework.classification.Classifier classifier = ctx.getPredictor();
        ctx.setEstimates(classifier.estimate(validationData));
        ctx.setPredictions(classifier.predict(validationData));
      }
    };
    validator.add(ClassifierEvaluator.getInstance());

    // This validator is also implemented here:
    Validator<Classifier> cvd = new org.mimirframework.classification.ClassifierValidator<>(partitioner);

    org.mimirframework.classification.LogisticRegression.Learner learner = new org.mimirframework.classification.LogisticRegression.Learner();
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris(), new Random(10));

    // Take all columns except 'Class', and replace any NA values with the mean of that column
    // as the training data
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Object.class, Is::NA), v.mean()));
    Vector y = iris.get("Class");

    // Test the classifier learner (in this class a logistic regression model)
    org.mimirframework.evaluation.Result result = validator.test(learner, x, y);

    // Get a data frame of measures (the ones computed by ZeroOneLossEvaluator and
    // ProbabilityEvaluator)
    DataFrame measures = result.getMeasures();
    System.out.println(measures);

    NearestNeighbours.Learner knn = new NearestNeighbours.Learner(3);
    org.mimirframework.evaluation.Result knnResult = validator.test(knn, x, y);
    System.out.println(knnResult.getMeasures().mean());

    System.out.println(measures.mean().sub(knnResult.getMeasures().mean()).abs());
  }
}
