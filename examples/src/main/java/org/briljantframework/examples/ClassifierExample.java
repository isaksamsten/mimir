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

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.ClassifierMeasure;
import org.briljantframework.mimir.classification.LogisticRegression;

/**
 * Created by isak on 11/18/15.
 */
public class ClassifierExample {

  public static void main(String[] args) {
    DataFrame iris = Datasets.loadIris();
    DataFrame x = iris.drop("Class");

    // Multiclass setting (3 classes), however, any c > 2 is supported
    Vector y = iris.get("Class");

    LogisticRegression.Learner lrl = new LogisticRegression.Learner();

    IntArray idx = Arrays.shuffle(Range.of(x.rows()));

    // Use the first 100 examples as training
    IntArray trainIdx = idx.get(Range.of(100));

    // And the last 50 examples as testing
    IntArray testIdx = idx.get(Range.of(100, 150));

    // Get the training data (replacing missing values with the column mean)
    DataFrame xTrain = x.values().getRecord(trainIdx).apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector yTrain = y.values().get(trainIdx);

    // Get the testing data
    DataFrame xTest = x.values().getRecord(testIdx).apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector yTest = y.values().get(testIdx);

    // Fit the logistic regression model
    LogisticRegression lr = lrl.fit(xTrain, yTrain);

    // Predict the class label of the test instances
    Vector predicted = lr.predict(xTest);

    // Compute the probabilities for each class
    DoubleArray scores = lr.estimate(xTest);

    // Compute some classifier measures (accuracy, auc, etc.)
    ClassifierMeasure cm = new ClassifierMeasure(predicted, yTest, scores, lr.getClasses());

    System.out.println(cm.getAccuracy());
    System.out.println(cm.getAreaUnderRocCurve());
    System.out.println(cm.getBrierScore());

    // We can also inspect the parameters
    System.out.println(lr.getOddsRatio("Sepal Width"));
    System.out.println(lr.getLogLoss());
    System.out.println(lr.getParameters());
  }
}
