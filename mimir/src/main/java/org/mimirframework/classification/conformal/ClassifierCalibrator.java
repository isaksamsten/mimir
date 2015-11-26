package org.mimirframework.classification.conformal;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.primitive.DoubleList;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface ClassifierCalibrator {

  /**
   * Returns an unconditional classifier calibrator
   * 
   * @return an unconditional classifier calibrator
   */
  static ClassifierCalibrator unconditional() {
    return (nc, x, y) -> {
      DoubleArray calibration = nc.estimate(x, y);
      return (example, label) -> calibration;
    };
  }

  /**
   * Returns a classifier calibrator that computes calibration scores conditional on the class.
   * 
   * @return a class conditional classifier calibrator
   */
  static ClassifierCalibrator classConditional() {
    return (nc, x, y) -> {
      Map<Object, DoubleList> tmpClassNc = new HashMap<>();
      for (int i = 0; i < x.rows(); i++) {
        Vector e = x.loc().getRecord(i);
        Object c = y.loc().get(i);
        DoubleList l = tmpClassNc.get(c);
        if (l == null) {
          l = new DoubleList();
          tmpClassNc.put(c, l);
        }
        l.add(nc.estimate(e, c));
      }
      Map<Object, DoubleArray> classNc =
          tmpClassNc.entrySet().stream()
              .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().toDoubleArray()));
      return (example, label) -> classNc.get(label);

    };
  }

  /**
   * @param nc the nonconformity score used to perform the calibration
   * @param x the calibration data
   * @param y the calibration target
   */
  ClassifierCalibratorScores calibrate(ClassifierNonconformity nc, DataFrame x, Vector y);
}
