package org.mimirframework.classification.conformal;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface ClassifierCalibratorScores {

  /**
   * Return a double array of [no-calibration, 1] double array of nonconformity scores for the
   * calibration set used to estimate the p-values when performing a conformal prediction. The
   * calibration set might be different for different for different examples or lables.
   *
   * @param example the example
   * @param label the label
   * @return the calibration scores
   */
  DoubleArray getCalibrationScores(Vector example, Object label);

}
