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

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;
import org.briljantframework.primitive.DoubleList;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface ClassifierCalibrator<In> {

  /**
   * Returns an unconditional classifier calibrator
   * 
   * @return an unconditional classifier calibrator
   */
  static <In> ClassifierCalibrator<In> unconditional() {
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
  static <In> ClassifierCalibrator<In> classConditional() {
    return (nc, x, y) -> {
      Map<Object, DoubleList> tmpClassNc = new HashMap<>();
      for (int i = 0; i < x.size(); i++) {
        In e = x.get(i);
        Object c = y.get(i);
        DoubleList l = tmpClassNc.get(c);
        if (l == null) {
          l = new DoubleList();
          tmpClassNc.put(c, l);
        }
        l.add(nc.estimate(e, c));
      }
      Map<Object, DoubleArray> classNc = tmpClassNc.entrySet().stream()
          .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().toDoubleArray()));
      return (example, label) -> classNc.get(label);

    };
  }

  /**
   * @param nc the nonconformity score used to perform the calibration
   * @param x the calibration data
   * @param y the calibration target
   */
  ClassifierCalibratorScores<In> calibrate(ClassifierNonconformity<In, Object> nc,
      Input<? extends In> x, Output<?> y);
}
