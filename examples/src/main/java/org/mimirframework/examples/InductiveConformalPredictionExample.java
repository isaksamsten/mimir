package org.mimirframework.examples;

import org.briljantframework.array.ArrayPrinter;
import org.briljantframework.array.Arrays;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.classification.conformal.ConformalClassifier;
import org.mimirframework.classification.conformal.InductiveConformalClassifier;
import org.mimirframework.classification.conformal.Nonconformity;
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
    ArrayPrinter.setMinimumTruncateSize(10000);
    DataFrame sc = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = sc.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = sc.get("Class");

    Predictor.Learner<? extends Classifier> classifier = // new LogisticRegression.Learner();
        new RandomForest.Configurator(100).configure();

    Nonconformity.Learner nc = new ProbabilityEstimateNonconformity.Learner(classifier,
        ProbabilityCostFunction.inverseProbability());
    InductiveConformalClassifier.Learner cp = new InductiveConformalClassifier.Learner(nc);
    Validator<ConformalClassifier> validator =
        ConformalClassifierValidator.crossValidator(10, 0.3, Arrays.linspace(0.01, 0.1, 9));
    Result result = validator.test(cp, x, y);
    System.out.println(result.getFitTime());
    System.out.println(
        result.getMeasures().groupBy("significance").collect(Vector::mean).sort(SortOrder.ASC));
  }
}
