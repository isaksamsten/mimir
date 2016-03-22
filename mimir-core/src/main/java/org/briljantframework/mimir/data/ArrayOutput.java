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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * A mutable set of arbitrary outputs.
 * 
 * @author Isak Karlsson
 */
public class ArrayOutput<T> extends AbstractOutput<T> {

  private final ArrayList<T> values = new ArrayList<>();

  /**
   * Creates a new output based on the given collection.
   * 
   * @param collection the collection of values.
   */
  public ArrayOutput(Collection<? extends T> collection) {
    values.addAll(collection);
  }

  /**
   * Creates a new output with initial capacity.
   */
  public ArrayOutput() {}

  public void clear() {
    values.clear();
  }

  public boolean addAll(Collection<? extends T> c) {
    return values.addAll(c);
  }

  public boolean addAll(int index, Collection<? extends T> c) {
    return values.addAll(index, c);
  }

  public T remove(int index) {
    return values.remove(index);
  }

  public boolean remove(Object o) {
    return values.remove(o);
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    return values.containsAll(c);
  }

  public void add(int index, T element) {
    values.add(index, element);
  }

  public T set(int index, T element) {
    return values.set(index, element);
  }

  public boolean removeAll(Collection<?> c) {
    return values.removeAll(c);
  }

  public boolean retainAll(Collection<?> c) {
    return values.retainAll(c);
  }

  public boolean isEmpty() {
    return values.isEmpty();
  }

  public boolean contains(Object o) {
    return values.contains(o);
  }

  @Override
  public Iterator<T> iterator() {
    return values.iterator();
  }

  @Override
  public Object[] toArray() {
    return values.toArray();
  }

  @Override
  public <E> E[] toArray(E[] a) {
    return values.toArray(a);
  }

  @Override
  public boolean add(T t) {
    return values.add(t);
  }

  public int indexOf(Object o) {
    return values.indexOf(o);
  }

  public int lastIndexOf(Object o) {
    return values.lastIndexOf(o);
  }

  @Override
  public T get(int index) {
    return values.get(index);
  }

  @Override
  public int size() {
    return values.size();
  }
}
