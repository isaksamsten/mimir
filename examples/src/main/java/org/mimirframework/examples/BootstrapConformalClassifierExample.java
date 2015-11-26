package org.mimirframework.examples;

import org.briljantframework.array.ArrayPrinter;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.classification.conformal.BootstrapConformalClassifier;
import org.mimirframework.classification.conformal.ProbabilityCostFunction;
import org.mimirframework.classification.conformal.evaluation.ConformalClassifierValidator;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class BootstrapConformalClassifierExample {
  public static void main(String[] args) {
    ArrayPrinter.setMinimumTruncateSize(100000);
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    Predictor.Learner<BootstrapConformalClassifier> learner =
        new BootstrapConformalClassifier.Learner(new RandomForest.Learner(100),
            ProbabilityCostFunction.margin());

    Validator<BootstrapConformalClassifier> cv =
        ConformalClassifierValidator.crossValidator(10, DoubleArray.range(0.01, 0.11, 0.01));
    Result result = cv.test(learner, x, y);
    DataFrame significance =
        result.getMeasures().groupBy(Double.class, v -> String.format("%.2f", v), "significance")
            .collect(Vector::mean);
    System.out.println(significance.select("error", "noClasses").sort(SortOrder.ASC));

  }
}
