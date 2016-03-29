package org.briljantframework.mimir.data;

import java.util.Arrays;

import org.briljantframework.data.vector.Convert;

/**
 * Created by isak on 3/29/16.
 */
class ImmutableArrayInstance implements Instance {
  private final Object[] array;

  ImmutableArrayInstance(Object[] array) {
    this.array = Arrays.copyOf(array, array.length);
  }

  @Override
  public int size() {
    return array.length;
  }

  @Override
  public double getAsDouble(int index) {
    return Convert.to(Double.class, index);
  }

  @Override
  public Object get(int index) {
    return array[index];
  }
}
