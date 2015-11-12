package org.briljantframework.evaluation;

import org.briljantframework.data.Na;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public enum PredictorMeasure {
  /**
   * Measures the time (in milliseconds) required to fit a predictor
   */
  FIT_TIME,

  /**
   * Measures the time (in milliseconds) required to make predictions using the predictor
   */
  PREDICT_TIME,

  /**
   * Measures the training set size
   */
  TRAINING_SIZE,

  /**
   * Measures the validation set size
   */
  VALIDATION_SIZE;
}
