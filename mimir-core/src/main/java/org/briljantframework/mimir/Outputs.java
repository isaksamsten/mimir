package org.briljantframework.mimir;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

import org.briljantframework.data.vector.Vector;

/**
 * Utilities for output variables
 */
public final class Outputs {

  private Outputs() {}

  /**
   * Returns an unmodifiable view of the specified output.
   *
   * @param output the output
   * @param <T> the class of the output
   * @return an unmodifiable view of the specified output
   */
  public static <T> Output<T> unmodifiableOutput(Output<? extends T> output) {
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
  public static <T> Output<T> newOutput(T... values) {
    return new ArrayOutput<>(values);
  }

  /**
   * Returns a new immutable output using the specified vector as a backing data structure.
   * 
   * @param cls the output type
   * @param vector the vector
   * @param <T> the class of outputs
   * @return a new ummutable output.
   */
  public static <T> Output<T> newOutput(Class<T> cls, Vector vector) {
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
  public static Output<?> newOutput(Vector vector) {
    return newOutput(Object.class, vector);
  }

  /**
   * Returns a new immutable output of real values.
   * 
   * @param vector the vector
   * @return a new immutable double output
   */
  public static Output<Double> newDoubleOutput(Vector vector) {
    Objects.requireNonNull(vector);
    return new VectorOutput<Double>(vector, Double.class) {
      @Override
      public Double get(int index) {
        return this.vector.loc().getAsDouble(index);
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
  public static <T> List<T> unique(Output<? extends T> output) {
    return new ArrayList<>(new HashSet<>(output));
  }

  /**
   * Returns a map with the counts of each distinct output value.
   * 
   * @param output the output
   * @return a new map with value counts
   */
  public static Map<Object, Integer> valueCounts(Output<?> output) {
    Map<Object, Integer> counts = new HashMap<>();
    for (int i = 0; i < output.size(); i++) {
      counts.compute(output.get(i), (k, v) -> v == null ? 1 : v + 1);
    }
    return counts;
  }

  private static class ArrayOutput<T> extends AbstractCollection<T> implements Output<T> {
    private final T[] values;

    public ArrayOutput(T[] values) {
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
    protected final Vector vector;
    protected final Class<T> cls;

    public VectorOutput(Vector vector, Class<T> cls) {
      this.vector = vector;
      this.cls = cls;
    }

    @Override
    public int size() {
      return vector.size();
    }

    @Override
    public T get(int index) {
      return vector.loc().get(cls, index);
    }

    @Override
    public Stream<T> stream() {
      return vector.stream(cls);
    }

    @Override
    public Stream<T> parallelStream() {
      return vector.stream(cls);
    }

    @Override
    public Iterator<T> iterator() {
      return vector.toList(cls).iterator();
    }
  }
}
