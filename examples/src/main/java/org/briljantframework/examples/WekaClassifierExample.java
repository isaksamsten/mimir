package org.briljantframework.examples;

import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.conformal.ClassifierNonconformity;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ProbabilityCostFunction;
import org.briljantframework.mimir.classification.conformal.ProbabilityEstimateNonconformity;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.supervised.Predictor;
import org.briljantframework.mimir.weka.WekaClassifier;

import weka.classifiers.trees.RandomForest;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class WekaClassifierExample {

  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    // Construct a weka classifier
    Predictor.Learner<? extends Classifier> classifier =
        new WekaClassifier.Learner<>(new RandomForest());

    // Use the classifier as a probability estimator
    ClassifierNonconformity.Learner nc =
        new ProbabilityEstimateNonconformity.Learner(classifier, ProbabilityCostFunction.margin());

    // Construct an inductive conformal classifier
    InductiveConformalClassifier.Learner cp = new InductiveConformalClassifier.Learner(nc);

    // and a suitable validator
    Validator<InductiveConformalClassifier> validator =
        ConformalClassifierValidator.crossValidator(10, 0.2);

    // Evaluate the conformal classifier
    Result result = validator.test(cp, x, y);

    // Compute the mean of all measures grouped by significance level
    DataFrame meanPerSignificance =
        result.getMeasures().groupBy("significance").collect(Vector::mean).sort(SortOrder.ASC);
    System.out.println(meanPerSignificance);
  }
}
