package org.mimirframework.classification.conformal;

import org.briljantframework.array.ArrayPrinter;
import org.briljantframework.array.Arrays;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifierTest {

  @Test
  public void testConformalPredictions() throws Exception {
    ArrayPrinter.setMinimumTruncateSize(10000);
    DataFrame sc = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = sc.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = sc.get("Class");

    org.mimirframework.supervised.Predictor.Learner<? extends Classifier> classifier = // new LogisticRegression.Learner();
        new org.mimirframework.classification.RandomForest.Configurator(100).configure();

    Nonconformity.Learner nc = new ProbabilityEstimateNonconformity.Learner(classifier,
        ProbabilityCostFunction.inverseProbability());
    InductiveConformalClassifier.Learner cp = new InductiveConformalClassifier.Learner(nc);
    org.mimirframework.evaluation.Validator<ConformalClassifier> validator =
        ConformalClassifierValidator.crossValidator(10, 0.3, Arrays.linspace(0.01, 0.1, 9));
    org.mimirframework.evaluation.Result result = validator.test(cp, x, y);
    System.out.println(result.getFitTime());
    System.out.println(
        result.getMeasures().groupBy("significance").collect(Vector::mean).sort(SortOrder.ASC));

    // EvaluationContextImpl context = new EvaluationContextImpl();
    // context.setPartition(test);
    // context.setPredictor(predictor);

    // new ConformalClassifierEvaluator(0.05).accept(context);
    // System.out.println(context.getMeasures());
  }

}
