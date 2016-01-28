/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
/**
 * This copy of Woodstox XML processor is licensed under the
 * Apache (Software) License, version 2.0 ("the License").
 * See the License for details about distribution rights, and the
 * specific rights regarding derivate works.
 *
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/
 *
 * A copy is also included in the downloadable source code package
 * containing Woodstox, in file "ASL2.0", under the same directory
 * as this file.
 */
package org.briljantframework.examples;

import java.util.List;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.ClassifierEvaluator;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.LogisticRegression;
import org.briljantframework.mimir.classification.tune.Configuration;
import org.briljantframework.mimir.classification.tune.GridSearch;
import org.briljantframework.mimir.classification.tune.Tuner;
import org.briljantframework.mimir.classification.tune.Updaters;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.evaluation.partition.FoldPartitioner;

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
    tuner.setParameter("iterations",
        Updaters.enumeration(LogisticRegression.Configurator::setIterations, 100, 200, 300))
        .setParameter("lambda",
            Updaters.linspace(LogisticRegression.Configurator::setRegularization, -10, 10.0, 10));

    // Get a list of configurations
    List<Configuration<LogisticRegression>> tune =
        tuner.tune(new LogisticRegression.Configurator(100), x, y);

    // Sort the configurations according the the mean accuracy
    tune.sort((a, b) -> -Double.compare(a.getResult().getMeasure("accuracy").mean(), b.getResult()
        .getMeasure("accuracy").mean()));

    for (Configuration<LogisticRegression> configuration : tune) {
      System.out.println(configuration.getParameters());
      System.out.println(configuration.getResult().getMeasures().mean());
    }
  }
}
