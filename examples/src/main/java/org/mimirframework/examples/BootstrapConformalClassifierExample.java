package org.mimirframework.examples;

import org.briljantframework.array.ArrayPrinter;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
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

    // IntArray idx = Arrays.shuffle(Range.of(iris.rows()));
    // IntArray train = idx.get(Range.of(0, 50));
    // IntArray cal = idx.get(Range.of(50, 100));
    // IntArray test = idx.get(Range.of(100, 150));
    //
    // RandomForest randomForest = new RandomForest();
    // randomForest.setNumTrees(100);
    // ProbabilityEstimateNonconformity.Learner nc =
    // new ProbabilityEstimateNonconformity.Learner(new
    // org.mimirframework.classification.RandomForest.Learner(100),
    // ProbabilityCostFunction.margin());
    // InductiveConformalClassifier.Learner c = new InductiveConformalClassifier.Learner(nc);
    // InductiveConformalClassifier icp = c.fit(x.loc().getRecord(train), y.loc().get(train));
    // icp.calibrate(x.loc().getRecord(cal), y.loc().get(cal));
    //
    // DoubleArray prediction = icp.estimate(x.loc().getRecord(test));
    // ConformalClassifierMeasure m =
    // new ConformalClassifierMeasure(y.loc().get(test), prediction, 0.95, icp.getClasses());
    // System.out.println(m.getError());
    Predictor.Learner<BootstrapConformalClassifier> learner =
        new BootstrapConformalClassifier.Learner(
            new org.mimirframework.classification.RandomForest.Learner(100),
            ProbabilityCostFunction.margin());

    Validator<BootstrapConformalClassifier> cv =
        ConformalClassifierValidator.crossValidator(10, DoubleArray.range(0.05, 1.01, 0.1));
    Result result = cv.test(learner, x, y);
    DataFrame significance =
        result.getMeasures().groupBy(Double.class, v -> String.format("%.2f", v), "significance")
            .collect(Vector::mean);
    System.out.println(DataFrames.toString(
        significance.sort(SortOrder.ASC).get("error", new Object[] {"noClasses"}), 10000));
  }
}
