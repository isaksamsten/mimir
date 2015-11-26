package org.mimirframework.examples;

import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.classification.conformal.ClassifierCalibrator;
import org.mimirframework.classification.conformal.ClassifierNonconformity;
import org.mimirframework.classification.conformal.InductiveConformalClassifier;
import org.mimirframework.classification.conformal.ProbabilityCostFunction;
import org.mimirframework.classification.conformal.ProbabilityEstimateNonconformity;
import org.mimirframework.classification.conformal.evaluation.ConformalClassifierValidator;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class InductiveConformalPredictionExample {
  public static void main(String[] args) {
    // Load the iris data set
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());

    // Remove the class variable from the input data and set each NA value to the column mean
    DataFrame x = iris.drop("Class");// .apply(v -> v.set(v.where(Is::NA), v.mean()));

    // Get the class variable
    Vector y = iris.get("Class");

    // Create a classifier learner to use for estimating the non-conformity scores
    Predictor.Learner<? extends Classifier> classifier =
        new RandomForest.Configurator(100).configure();

    // Initialize the non-conformity learner using the margin as cost function
    ClassifierNonconformity.Learner nc =
        new ProbabilityEstimateNonconformity.Learner(classifier, ProbabilityCostFunction.margin());

    // Initialize an inductive conformal classifier using the non-conformity learner
    // and calibrator
    ClassifierCalibrator calibrator = ClassifierCalibrator.unconditional();
    InductiveConformalClassifier.Learner cp =
        new InductiveConformalClassifier.Learner(nc, calibrator);

    // Creates a validator for evaluating the validity and efficiency of the conformal classifier.
    // In this case, we evaluate the classifier using 10-fold cross-validation and 10 significance
    // levels between 0.1 and 0.1
    Validator<InductiveConformalClassifier> validator =
        ConformalClassifierValidator.crossValidator(10, 0.25);

    Result result = validator.test(cp, x, y);

    // Get the measures
    DataFrame measures = result.getMeasures();

    // Compute the mean of all measures grouped by significance level
    // Only selecting the significance level, error rate and number of classes
    DataFrame meanPerSignificance =
        measures.select("significance", "error", "noClasses")
            .groupBy(Double.class, v -> String.format("%.2f", v), "significance")
            .collect(Vector::mean).sort(SortOrder.ASC);

    System.out.println(meanPerSignificance);
  }
}
