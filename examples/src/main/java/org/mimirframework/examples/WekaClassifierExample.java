package org.mimirframework.examples;

import org.briljantframework.array.Arrays;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.conformal.InductiveConformalClassifier;
import org.mimirframework.classification.conformal.Nonconformity;
import org.mimirframework.classification.conformal.ProbabilityCostFunction;
import org.mimirframework.classification.conformal.ProbabilityEstimateNonconformity;
import org.mimirframework.classification.conformal.evaluation.ConformalClassifierValidator;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.supervised.Predictor;
import org.mimirframework.weka.WekaClassifier;

import weka.classifiers.trees.RandomForest;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class WekaClassifierExample {

  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");
    Predictor.Learner<? extends Classifier> classifier =
        new WekaClassifier.Learner<>(new RandomForest());
    Nonconformity.Learner nc =
        new ProbabilityEstimateNonconformity.Learner(classifier, ProbabilityCostFunction.margin());
    InductiveConformalClassifier.Learner cp = new InductiveConformalClassifier.Learner(nc);
    Validator<InductiveConformalClassifier> validator =
        ConformalClassifierValidator.crossValidator(10, 0.3);
    Result result = validator.test(cp, x, y);
    DataFrame measures = result.getMeasures();

    // Compute the mean of all measures grouped by significance level
    DataFrame meanPerSignificance =
        measures.groupBy("significance").collect(Vector::mean)
            .sort((a, b) -> Double.compare((Double) a, (Double) b));
    System.out.println(meanPerSignificance);
  }
}
