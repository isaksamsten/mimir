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
package org.briljantframework.mimir.evaluation;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.Output;

/**
 * @author Isak Karlsson
 */
public class Result<T> {

  private final double trainingSize;
  private final double validationSize;
  private final double fitTime;
  private final double predictTime;

  private final DataFrame measures;
  private final Output<T> predictions;
  private final Output<T> actual;

  public Result(MeasureCollection measures, Output<T> t, Output<T> p, double trainingSize,
      double validationSize, double fitTime, double predictTime) {
    this.trainingSize = trainingSize;
    this.validationSize = validationSize;
    this.fitTime = fitTime;
    this.predictTime = predictTime;
    this.measures = measures.toDataFrame();
    this.actual = t;
    this.predictions = p;
  }

  /**
   * Returns the size of the training set.
   *
   * @return the size of the training set
   */
  public double getTrainingSize() {
    return trainingSize;
  }

  /**
   * Returns the size of the validation set.
   * 
   * @return the size of the validation set
   */
  public double getValidationSize() {
    return validationSize;
  }

  /**
   * Return the time it took to fit the model (in milliseconds)
   * 
   * @return the time it took to fit the model
   */
  public double getFitTime() {
    return fitTime;
  }

  /**
   * Return the time it took to use the model for prediction (in milliseconds)
   * 
   * @return the time it took to use the model for prediction
   */
  public double getPredictTime() {
    return predictTime;
  }

  /**
   * Get a data frame of measurements where each column is a measurement and each row an evaluation
   * 
   * @return a data frame of measures
   */
  public DataFrame getMeasures() {
    return measures;
  }

  /**
   * Get the values for a specific measurement
   * 
   * @param measure the measurement
   * @return a vector of measurements
   */
  public Series getMeasure(String measure) {
    return measures.get(measure);
  }

  /**
   * Get a vector of predictions
   * 
   * @return the vector of predictions
   */
  public Output<T> getPredictions() {
    return predictions;
  }

  /**
   * Get the actual values
   * 
   * @return a vector of actual values
   */
  public Output<T> getActual() {
    return actual;
  }

  @Override
  public String toString() {
    return getMeasures().toString();
  }
}
