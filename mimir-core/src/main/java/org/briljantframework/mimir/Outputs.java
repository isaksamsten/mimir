package org.briljantframework.mimir;

import java.util.*;

import org.briljantframework.array.DoubleArray;

/**
 * Utilities for output variables
 */
public final class Outputs {

  private Outputs() {}

  @SafeVarargs
  @SuppressWarnings("varargs")
  public static <T> Output<T> asOutput(T... values) {
    return new UnmodifiableArrayOutput<>(values);
  }

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

  private static class UnmodifiableArrayOutput<T> extends AbstractCollection<T>
      implements Output<T> {
    private final T[] values;

    public UnmodifiableArrayOutput(T[] values) {
      this.values = values;
    }

    @Override
    public Iterator<T> iterator() {
      return new Iterator<T>() {
        private int current = 0;

        @Override
        public boolean hasNext() {
          return current < size();
        }

        @Override
        public T next() {
          return get(current++);
        }
      };
    }

    @Override
    public int size() {
      return values.length;
    }

    @Override
    public T get(int i) {
      return values[i];
    }
  }
}
