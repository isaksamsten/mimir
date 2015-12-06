package org.briljantframework.mimir.classification.conformal;

import java.util.stream.IntStream;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface ClassifierNonconformity {

  /**
   * Estimate the nonconformity score for each example (record) in the given dataframe w.r.t. to the
   * true class label in y
   *
   * @param x the given dataframe of examples
   * @param y the true class labels of the examples
   * @return a {@code [no examples]} double array of nonconformity scores
   */
  default DoubleArray estimate(DataFrame x, Vector y) {
    DoubleArray array = DoubleArray.zeros(x.rows());
    // Run in parallel
    IntStream.range(0, x.rows()).parallel()
        .forEach(i -> array.set(i, estimate(x.loc().getRecord(i), y.loc().get(i))));
    return array;
  }

  /**
   * Estimate the nonconformity score for the given example and label
   * 
   * @param example the given example
   * @param label the given label
   * @return the nonconformity score of example w.r.t the
   */
  double estimate(Vector example, Object label);

  /**
   * Get the classes used by this nonconformity scorer
   * 
   * @return the classes
   */
  Vector getClasses();

  /**
   * Learn a {@linkplain ClassifierNonconformity nonconformity score function} using the given data
   * and targets.
   *
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  interface Learner {

    /**
     * Fit a {@linkplain ClassifierNonconformity nonconformity score function} using the given data.
     *
     * @param x the input data
     * @param y the input target
     * @return a nonconformity score function
     */
    ClassifierNonconformity fit(DataFrame x, Vector y);
  }
}
