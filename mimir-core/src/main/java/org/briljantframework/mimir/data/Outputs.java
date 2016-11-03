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
package org.briljantframework.mimir.data;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

import org.briljantframework.DoubleSequence;
import org.briljantframework.array.Array;
import org.briljantframework.data.series.Convert;
import org.briljantframework.data.series.Series;

/**
 * Utilities for output variables
 */
public final class Outputs {

  private Outputs() {}

  public static Output<Double> copyOf(DoubleSequence sequence) {
    Output<Double> output = new ArrayOutput<>();
    for (int i = 0; i < sequence.size(); i++) {
      output.add(sequence.getDouble(i));
    }
    return unmodifiableOutput(output);
  }

  /**
   * Returns an unmodifiable view of the specified output.
   *
   * @param output the output
   * @param <T> the class of the output
   * @return an unmodifiable view of the specified output
   */
  @SuppressWarnings("unchecked")
  public static <T> Output<T> unmodifiableOutput(Output<? extends T> output) {
    if (output instanceof UnmodifiableOutput) {
      return (Output<T>) output;
    }
    return new UnmodifiableOutput<>(output);
  }

  /**
   * Returns a new unmodifiable output. Changes to the input array propagates to the output.
   * 
   * @param values the value array
   * @param <T> the class of the output
   * @return a new unmodifiable output
   */
  @SafeVarargs
  @SuppressWarnings("varargs")
  public static <T> Output<T> asOutput(T... values) {
    return new ImmutableArrayOutput<>(values);
  }

  /**
   * Returns a new immutable output using the specified vector as a backing data structure.
   * 
   * @param cls the output type
   * @param vector the vector
   * @param <T> the class of outputs
   * @return a new ummutable output.
   */
  public static <T> Output<T> asOutput(Class<T> cls, Series vector) {
    Objects.requireNonNull(vector);
    Objects.requireNonNull(cls, "illegal class");
    return new VectorOutput<>(vector, cls);
  }

  /**
   * Returns a new immutable output using the specified vector.
   * 
   * @param vector the vector
   * @return a new immutable output.
   */
  public static Output<?> asOutput(Series vector) {
    return asOutput(Object.class, vector);
  }

  /**
   * Returns a new immutable output of real values.
   * 
   * @param vector the vector
   * @return a new immutable double output
   */
  public static Output<Double> newDoubleOutput(Series vector) {
    Objects.requireNonNull(vector);
    return new VectorOutput<Double>(vector, Double.class) {
      @Override
      public Double get(int index) {
        return this.vector.values().getDouble(index);
      }
    };
  }

  /**
   * Returns a list of unique output values.
   * 
   * @param output the output
   * @param <T> the class of elements
   * @return the new list of unique output values.
   */
  public static <T> Array<T> unique(Output<? extends T> output) {
    return Array.copyOf(new HashSet<>(output));
  }

  public static <T> Array<T> unique(Collection<? extends Output<? extends T>> outputs) {
    HashSet<T> set = new HashSet<>();
    for (Output<? extends T> output : outputs) {
      set.addAll(output);
    }
    return Array.copyOf(set);
  }

  /**
   * Returns a map with the counts of each distinct output value.
   * 
   * @param output the output
   * @return a new map with value counts
   */
  public static <T> Map<T, Integer> valueCounts(Output<? extends T> output) {
    Map<T, Integer> counts = new HashMap<>();
    for (T value : output) {
      counts.compute(value, (k, v) -> v == null ? 1 : v + 1);
    }
    return counts;
  }

  private static class ImmutableArrayOutput<T> extends AbstractList<T> implements Output<T> {
    private final T[] values;

    public ImmutableArrayOutput(T[] values) {
      this.values = Objects.requireNonNull(values);
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
    public T get(int index) {
      return values[index];
    }
  }


  private static class UnmodifiableOutput<E> extends AbstractOutput<E> {
    private final Output<? extends E> c;

    public UnmodifiableOutput(Output<? extends E> c) {
      this.c = Objects.requireNonNull(c);
    }

    public int size() {
      return c.size();
    }

    public boolean isEmpty() {
      return c.isEmpty();
    }

    public boolean contains(Object o) {
      return c.contains(o);
    }

    public Object[] toArray() {
      return c.toArray();
    }

    public <T> T[] toArray(T[] a) {
      return c.toArray(a);
    }

    public String toString() {
      return c.toString();
    }

    public Iterator<E> iterator() {
      return new Iterator<E>() {
        private final Iterator<? extends E> i = c.iterator();

        public boolean hasNext() {
          return i.hasNext();
        }

        public E next() {
          return i.next();
        }

        public void remove() {
          throw new UnsupportedOperationException();
        }

        @Override
        public void forEachRemaining(Consumer<? super E> action) {
          // Use backing collection version
          i.forEachRemaining(action);
        }
      };
    }

    public boolean add(E e) {
      throw new UnsupportedOperationException();
    }

    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    public boolean containsAll(Collection<?> coll) {
      return c.containsAll(coll);
    }

    public boolean addAll(Collection<? extends E> coll) {
      throw new UnsupportedOperationException();
    }

    public boolean removeAll(Collection<?> coll) {
      throw new UnsupportedOperationException();
    }

    public boolean retainAll(Collection<?> coll) {
      throw new UnsupportedOperationException();
    }

    public void clear() {
      throw new UnsupportedOperationException();
    }

    // Override default methods in Collection
    @Override
    public void forEach(Consumer<? super E> action) {
      c.forEach(action);
    }

    @Override
    public boolean removeIf(Predicate<? super E> filter) {
      throw new UnsupportedOperationException();
    }

    @SuppressWarnings("unchecked")
    @Override
    public Spliterator<E> spliterator() {
      return (Spliterator<E>) c.spliterator();
    }

    @SuppressWarnings("unchecked")
    @Override
    public Stream<E> stream() {
      return (Stream<E>) c.stream();
    }

    @SuppressWarnings("unchecked")
    @Override
    public Stream<E> parallelStream() {
      return (Stream<E>) c.parallelStream();
    }

    @Override
    public E get(int index) {
      return c.get(index);
    }
  }


  private static class VectorOutput<T> extends AbstractOutput<T> {
    protected final Series vector;
    protected final Class<T> cls;

    VectorOutput(Series vector, Class<T> cls) {
      this.vector = vector;
      this.cls = cls;
    }

    @Override
    public int size() {
      return vector.size();
    }

    @Override
    public T get(int index) {
      return vector.values().get(cls, index);
    }

    @Override
    public Stream<T> stream() {
      return vector.values().stream().map(o -> Convert.to(cls, o));
    }

    @Override
    public Stream<T> parallelStream() {
      return vector.values().parallelStream().map(o -> Convert.to(cls, o));
    }

    @Override
    public Iterator<T> iterator() {
      return vector.values(cls).iterator();
    }
  }
}
