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


import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.mimir.supervised.data.Instance;

import java.util.Collection;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * @author Isak Karlsson
 */
public final class Inputs {

  private Inputs() {}

  /**
   * Returns an unmodifiable view of the given input. A shallow copy of the input properties are
   * constructed, i.e., mutable collections stored in the properties can be mutated through the
   * view. Similarly, mutations to the original propagates to the unmodifiable view.
   * 
   * @param input the input
   * @return an unmodifiable view of the given input
   */
  @SuppressWarnings("unchecked")
  public static <T> Input<T> unmodifiableInput(Input<? extends T> input) {
    if (input instanceof UnmodifiableInput) {
      return (Input<T>) input;
    }
    return new UnmodifiableInput<>(input);
  }


  private static class UnmodifiableInput<E> extends Input<E> {
    private final Input<? extends E> c;

    UnmodifiableInput(Input<? extends E> c) {
      this.c = c;
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

    @Override
    @SuppressWarnings("unchecked")
    public Schema<E> getSchema() {
      return (Schema<E>) c.getSchema();
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
}
