package org.mimirframework.classification.tune;

import java.util.List;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.junit.Test;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.LogisticRegression;
import org.mimirframework.classification.evaluation.ClassifierEvaluator;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.evaluation.partition.FoldPartitioner;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class GridSearchTest {

  @Test
  public void testTuen() throws Exception {
    Validator<LogisticRegression> cv = new ClassifierValidator<>(new FoldPartitioner(10));
    cv.add(ClassifierEvaluator.getInstance());
    Tuner<LogisticRegression, LogisticRegression.Configurator> tuner = new GridSearch<>(cv);
    // @formatter:off
    tuner.setParameter("iterations", Updaters.enumeration(LogisticRegression.Configurator::setIterations, 100, 200, 300))
        .setParameter("lambda", Updaters.linspace(LogisticRegression.Configurator::setRegularization, -10, 10.0, 10));
    // @formatter:on
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris()).filter(v -> !v.hasNA());
    DataFrame x = iris.drop("Class");
    Vector y = iris.get("Class");
    List<Configuration<LogisticRegression>> tune =
        tuner.tune(new LogisticRegression.Configurator(100), x, y);

    tune.sort((a, b) -> -Double.compare(a.getResult().getMeasure("accuracy").mean(),
        b.getResult().getMeasure("accuracy").mean()));
    for (Configuration<LogisticRegression> configuration : tune) {
      System.out.println(configuration.getParameters());
      System.out.println(configuration.getResult().getMeasures().mean());
    }
  }
}
