package org.mimirframework.distance;

import java.util.Objects;

import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class HammingDistance implements Distance {

  @Override
  public double compute(double a, double b) {
    return a == b ? 0 : 1;
  }

  @Override
  public double compute(Vector a, Vector b) {
    int size = Math.min(a.size(), b.size());
    double distance = 0;
    for (int i = 0; i < size; i++) {
      Object av = a.loc().get(Object.class, i);
      Object bv = b.loc().get(Object.class, i);
      distance += Objects.equals(av, bv) ? 0 : 1;
    }

    return distance / size;
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double min() {
    return Double.NEGATIVE_INFINITY;
  }
}
