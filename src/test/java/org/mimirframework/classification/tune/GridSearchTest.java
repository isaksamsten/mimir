package org.mimirframework.classification.tune;

import static org.mimirframework.classification.tune.Updaters.linspace;

import java.util.List;

import org.mimirframework.classification.LogisticRegression;
import org.mimirframework.classifier.evaluation.ClassifierEvaluator;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.evaluation.Validator;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class GridSearchTest {

  @Test
  public void testTuen() throws Exception {
    Validator<org.mimirframework.classification.LogisticRegression> cv = new org.mimirframework.classification.ClassifierValidator<>(new org.mimirframework.evaluation.partition.FoldPartitioner(10));
    cv.add(ClassifierEvaluator.getInstance());
    Tuner<LogisticRegression, LogisticRegression.Configurator> tuner = new org.mimirframework.classification.tune.GridSearch<>(cv);
    // @formatter:off
    tuner.setParameter("iterations", Updaters.enumeration(org.mimirframework.classification.LogisticRegression.Configurator::setIterations, 100, 200, 300))
        .setParameter("lambda", Updaters.linspace(org.mimirframework.classification.LogisticRegression.Configurator::setRegularization, -10, 10.0, 10));
    // @formatter:on
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris()).filter(v -> !v.hasNA());
    DataFrame x = iris.drop("Class");
    Vector y = iris.get("Class");
    List<org.mimirframework.classification.tune.Configuration<LogisticRegression>> tune =
        tuner.tune(new org.mimirframework.classification.LogisticRegression.Configurator(100), x, y);

    tune.sort((a, b) -> -Double.compare(a.getResult().getMeasure("accuracy").mean(),
        b.getResult().getMeasure("accuracy").mean()));
    for (org.mimirframework.classification.tune.Configuration<LogisticRegression> configuration : tune) {
      System.out.println(configuration.getParameters());
      System.out.println(configuration.getResult().getMeasures().mean());
    }
  }
}
