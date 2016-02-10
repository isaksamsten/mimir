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
package org.briljantframework.mimir.classification.conformal.evaluation;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ConformalClassifierMeasure {

  private final double accuracy, error, averagePvalue, confidence, credibility, singletons,
      noClasses;

  public ConformalClassifierMeasure(Vector truth, DoubleArray score, double significance,
                                    Vector classes) {
    // Compute confidence and credibility
    double correct = 0;
    double avgConfidence = 0;
    double avgCredibility = 0;
    double noSingletons = 0;
    double avgPValue = 0;
    double avgNoClasses = 0;
    for (int i = 0; i < score.rows(); i++) {
      DoubleArray estimate = score.getRow(i);
      BooleanArray predictions = estimate.where(p -> p > significance);
      double noPredictions = Arrays.sum(predictions);

      int trueClassIndex = classes.loc().indexOf(truth.loc().get(i));
      // if the true class wasn't included during training, it can't be correct
      if (trueClassIndex < 0) {
        correct++;
      } else {
        correct += predictions.get(trueClassIndex) ? 1 : 0;
        if (noPredictions == 1 && predictions.get(trueClassIndex)) {
          noSingletons++;
        }
      }
      int prediction = Arrays.argmax(estimate);
      double credibility = estimate.get(prediction);
      double confidence = 1 - Arrays.maxExcluding(estimate, prediction);

      avgCredibility += credibility / score.rows();
      avgConfidence += confidence / score.rows();
      avgPValue += Arrays.mean(estimate) / score.rows();
      avgNoClasses += noPredictions / score.rows();
    }
    accuracy = correct / truth.size();
    error = 1 - accuracy;
    averagePvalue = avgPValue;
    confidence = avgConfidence;
    credibility = avgCredibility;
    singletons = noSingletons / truth.size();
    noClasses = avgNoClasses;
  }

  public double getAccuracy() {
    return accuracy;
  }

  public double getError() {
    return error;
  }

  public double getAveragePvalue() {
    return averagePvalue;
  }

  public double getConfidence() {
    return confidence;
  }

  public double getCredibility() {
    return credibility;
  }

  public double getSingletons() {
    return singletons;
  }

  public double getNoClasses() {
    return noClasses;
  }

  @Override
  public String toString() {
    return "ConformalClassifierMeasure{" + "accuracy=" + accuracy + ", error=" + error
        + ", averagePvalue=" + averagePvalue + ", confidence=" + confidence + ", credibility="
        + credibility + ", singletons=" + singletons + ", noClasses=" + noClasses + '}';
  }
}
