package org.mimirframework.examples;

import java.util.List;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.LogisticRegression;
import org.mimirframework.classification.evaluation.ClassifierEvaluator;
import org.mimirframework.classification.tune.Configuration;
import org.mimirframework.classification.tune.GridSearch;
import org.mimirframework.classification.tune.Tuner;
import org.mimirframework.classification.tune.Updaters;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.evaluation.partition.FoldPartitioner;

/**
 * @author Isak Karlsson
 */
public class GridSearchExample {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris()).filter(v -> !v.hasNA());
    DataFrame x = iris.drop("Class");
    Vector y = iris.get("Class");

    // Initialize a 10-fold cross validator
    Validator<LogisticRegression> cv = new ClassifierValidator<>(new FoldPartitioner(10));

    // Add some traditional classifier evaluation measures
    cv.add(ClassifierEvaluator.INSTANCE);

    // Create a grid search tuner that uses cross-validation to find the best parameters
    Tuner<LogisticRegression, LogisticRegression.Configurator> tuner = new GridSearch<>(cv);
    tuner
        .setParameter("iterations",
            Updaters.enumeration(LogisticRegression.Configurator::setIterations, 100, 200, 300))
        .setParameter("lambda",
            Updaters.linspace(LogisticRegression.Configurator::setRegularization, -10, 10.0, 10));

    // Get a list of configurations
    List<Configuration<LogisticRegression>> tune =
        tuner.tune(new LogisticRegression.Configurator(100), x, y);

    // Sort the configurations according the the mean accuracy
    tune.sort((a, b) -> -Double.compare(a.getResult().getMeasure("accuracy").mean(),
        b.getResult().getMeasure("accuracy").mean()));

    for (Configuration<LogisticRegression> configuration : tune) {
      System.out.println(configuration.getParameters());
      System.out.println(configuration.getResult().getMeasures().mean());
    }
  }
}
