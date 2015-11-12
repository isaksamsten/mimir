package org.mimirframework.classifier.conformal;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface Nonconformity {

  /**
   * Estimate the nonconformity score for each example (record) in the given dataframe w.r.t. to the
   * true class label in y
   * 
   * @param x the given dataframe of examples
   * @param y the true class labels of the examples
   * @return a {@code [no examples]} double array of nonconformity scores
   */
  DoubleArray estimate(DataFrame x, Vector y);

  /**
   * Estimate the nonconformity score for the given example and label
   * 
   * @param example the given example
   * @param label the given label
   * @return the nonconformity score of example w.r.t the
   */
  double estimate(Vector example, Object label);

  /**
   * Learn a {@linkplain Nonconformity nonconformity score function} using the given data and
   * targets.
   *
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  interface Learner {

    /**
     * Fit a {@linkplain Nonconformity nonconformity score function} using the given data.
     *
     * @param x the input data
     * @param y the input target
     * @return a nonconformity score function
     */
    Nonconformity fit(DataFrame x, Vector y);
  }
}
