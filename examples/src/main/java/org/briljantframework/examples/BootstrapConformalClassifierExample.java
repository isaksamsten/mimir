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
package org.briljantframework.examples;

import org.briljantframework.array.ArrayPrinter;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.classification.conformal.BootstrapConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ProbabilityCostFunction;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.supervised.Predictor;

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
