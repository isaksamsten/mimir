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
package org.briljantframework.mimir.classification.conformal;

import java.util.stream.IntStream;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Na;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.Classifier;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface ConformalClassifier extends Classifier {

  /**
   * Returns the conformal predictions for the records in the given data frame using the given
   * significance level.
   *
   * @param x to determine class labels for
   * @return a vector of class-labels for those records with which a label can be assigned with the
   *         given probability; or {@code NA}.
   */
  default Vector predict(DataFrame x, double significance) {
    Vector.Builder predictions = getClasses().newBuilder();
    for (int i = 0, size = x.rows(); i < size; i++) {
      predictions.add(predict(x.loc().getRecord(i), significance));
    }
    return predictions.build();
  }

  /**
   * Returns the prediction of the given example or {@code NA}.
   *
   * @param record to which the class label shall be assigned
   * @return a class-label or {@code NA}
   */
  @Override
  Object predict(Vector record);

  /**
   * Returns the prediction of the given example or {@code NA}. A prediction is given iff one class
   * have a significance greater than or equal to the specified significance level.
   *
   * @param record to which the class label shall be assigned
   * @return a class-label or {@code NA}
   */
  default Object predict(Vector record, double significance) {
    DoubleArray estimate = estimate(record);
    if (estimate.filter(v -> v > significance).size() == 1) {
      return getClasses().loc().get(Arrays.argmax(estimate));
    } else {
      return Na.of(getClasses().getType().getDataClass());
    }
  }

  /**
   * Returns a boolean array {@code [n-classes]}, where each element denotes which labels are
   * included in the prediction set.
   *
   * @param example the example to predict
   * @param significance the significance level
   * @return a boolean array
   */
  default BooleanArray conformalPredict(Vector example, double significance) {
    return estimate(example).where(v -> v >= significance);
  }

  /**
   * Returns a boolean array of {@code [no examples, no classes]}, where each element denotes whihc
   * labels are included in the prediction set for the i:th example
   * 
   * @param x the data frame
   * @param significance the specified significance
   * @return a boolean array
   */
  default BooleanArray conformalPredict(DataFrame x, double significance) {
    BooleanArray estimates = Arrays.booleanArray(x.rows(), getClasses().size());
    IntStream.range(0, x.rows()).parallel().forEach(i -> {
      BooleanArray estimate = conformalPredict(x.loc().getRecord(i), significance);
      estimates.setRow(i, estimate);
    });
    return estimates;
  }

  /**
   * Returns a vector of possible predictions with a significance greater than or equal to the
   * specified significance level.
   * 
   * @param example the given example
   * @param significance the given significance level
   * @return a vector of possible predictions
   */
  default Vector predictionSet(Vector example, double significance) {
    Vector.Builder set = getClasses().newBuilder();
    BooleanArray predict = conformalPredict(example, significance);
    for (int i = 0; i < predict.size(); i++) {
      if (predict.get(i)) {
        set.add(getClasses(), i);
      }
    }
    return set.build();
  }

  /**
   * Returns an {@code [n-samples, n-classes]} double array of p-values associated with each class.
   *
   * @param x the data frame of records to estimate the p-values
   * @return the p-values
   */
  @Override
  DoubleArray estimate(DataFrame x);

  @Override
  DoubleArray estimate(Vector record);
}
