package org.briljantframework.mimir;

import java.util.*;

import org.briljantframework.array.DoubleArray;

/**
 * Utilities for output variables
 */
public final class Outputs {

  private Outputs() {}

  public static <T> List<T> unique(Output<? extends T> output) {
    return new ArrayList<>(new HashSet<>(output));
  }

  public static Map<Object, Integer> valueCounts(Output<?> output) {
    Map<Object, Integer> counts = new HashMap<>();
    for (int i = 0; i < output.size(); i++) {
      counts.compute(output.get(i), (k, v) -> v == null ? 1 : v + 1);
    }
    return counts;
  }

  public static DoubleArray toDoubleArray(Output<? extends Number> y) {
    DoubleArray out = DoubleArray.zeros(y.size());
    int i = 0;
    for (Number v : y) {
      out.set(i++, v.doubleValue());
    }

    return out;
  }
}
