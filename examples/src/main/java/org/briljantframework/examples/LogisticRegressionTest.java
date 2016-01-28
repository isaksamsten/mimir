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

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.LogisticRegression;
import org.briljantframework.mimir.evaluation.Result;

/**
 * @author Isak Karlsson
 */
public class LogisticRegressionTest {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    // Multinomial logistic regression
    Classifier.Learner<LogisticRegression> classifier = new LogisticRegression.Learner();
    Result result = ClassifierValidator.crossValidator(10).test(classifier, x, y);
    DataFrame measures = result.getMeasures();
    System.out.println(measures.mean());

    testOdds();
  }

  public static void testOdds() {
    // Construct a dataset
    DataFrame x = DataFrame.of("Age", Vector.of(55, 28, 65, 46, 86, 56, 85, 33, 21, 42), "Smoker",
        Vector.of(0, 0, 1, 0, 1, 1, 0, 0, 1, 1));

    // Construct a target (got cancer / not)
    Vector y = Vector.of(0, 0, 0, 1, 1, 1, 0, 0, 0, 1);

    // Show the dataset
    System.out.println(x.set("Cancer?", y));

    LogisticRegression.Learner regression = new LogisticRegression.Learner();
    LogisticRegression model = regression.fit(x, y);

    // Print the model
    System.out.println(model);

    // Get the odds ratio for the parameters
    System.out.println("(Intercept) " + model.getOddsRatio("(Intercept)"));
    for (Object o : x.getColumnIndex()) {
      System.out.println(o + " " + model.getOddsRatio(o));
    }
  }
}
