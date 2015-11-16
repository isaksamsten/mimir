package org.mimirframework.examples;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.RandomShapeletForest;
import org.mimirframework.classification.conformal.BootstrapConformalClassifier;
import org.mimirframework.classification.conformal.ProbabilityCostFunction;
import org.mimirframework.classification.conformal.evaluation.ConformalClassifierValidator;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.supervised.Predictor;

/**
 * Created by isak on 11/16/15.
 */
public class BootstrapConformalClassifierExample {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadSyntheticControl());
    DataFrame x = iris.drop(0);
    Vector y = iris.get(0);

    Predictor.Learner<BootstrapConformalClassifier> learner =
        new BootstrapConformalClassifier.Learner(
            new RandomShapeletForest.Configurator(100).configure(),
            ProbabilityCostFunction.margin());

    Validator<BootstrapConformalClassifier> cv = ConformalClassifierValidator.crossValidator(10);
    Result result = cv.test(learner, x, y);
    System.out.println(result.getMeasures().groupBy("significance").collect(Vector::mean));
  }
}


