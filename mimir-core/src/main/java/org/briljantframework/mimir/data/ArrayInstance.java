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
import java.util.function.UnaryOperator;
import java.util.stream.Stream;

import org.briljantframework.data.series.Convert;

/**
 * @author Isak Karlsson
 */
public class ArrayInstance extends AbstractList<Object> implements Instance {

  private final List<Object> values;

  public ArrayInstance(Collection<?> values) {
    this.values = new ArrayList<>(values);
  }

  public ArrayInstance() {
    this.values = new ArrayList<>();
  }

  @Override
  public Spliterator<Object> spliterator() {
    return values.spliterator();
  }

  @Override
  public boolean isEmpty() {
    return values.isEmpty();
  }

  @Override
  public boolean contains(Object o) {
    return values.contains(o);
  }

  @Override
  public Iterator<Object> iterator() {
    return values.iterator();
  }

  @Override
  public Object[] toArray() {
    return values.toArray();
  }

  @Override
  public <T> T[] toArray(T[] a) {
    return values.toArray(a);
  }

  @Override
  public boolean add(Object o) {
    return values.add(o);
  }

  @Override
  public boolean remove(Object o) {
    return values.remove(o);
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    return values.containsAll(c);
  }

  @Override
  public boolean addAll(Collection<?> c) {
    return values.addAll(c);
  }

  @Override
  public boolean addAll(int index, Collection<?> c) {
    return values.addAll(index, c);
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    return values.removeAll(c);
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    return values.retainAll(c);
  }

  @Override
  public void replaceAll(UnaryOperator<Object> operator) {
    values.replaceAll(operator);
  }

  @Override
  public void sort(Comparator<? super Object> c) {
    values.sort(c);
  }

  @Override
  public void clear() {
    values.clear();
  }

  @Override
  public boolean equals(Object o) {
    return values.equals(o);
  }

  @Override
  public int hashCode() {
    return values.hashCode();
  }

  @Override
  public Object get(int index) {
    return values.get(index);
  }

  @Override
  public Object set(int index, Object element) {
    return values.set(index, element);
  }

  @Override
  public void add(int index, Object element) {
    values.add(index, element);
  }

  @Override
  public Object remove(int index) {
    return values.remove(index);
  }

  @Override
  public int indexOf(Object o) {
    return values.indexOf(o);
  }

  @Override
  public int lastIndexOf(Object o) {
    return values.lastIndexOf(o);
  }

  @Override
  public ListIterator<Object> listIterator() {
    return values.listIterator();
  }

  @Override
  public ListIterator<Object> listIterator(int index) {
    return values.listIterator(index);
  }

  @Override
  public List<Object> subList(int fromIndex, int toIndex) {
    return values.subList(fromIndex, toIndex);
  }

  @Override
  public boolean removeIf(Predicate<? super Object> filter) {
    return values.removeIf(filter);
  }

  @Override
  public Stream<Object> stream() {
    return values.stream();
  }

  @Override
  public Stream<Object> parallelStream() {
    return values.parallelStream();
  }

  @Override
  public void forEach(Consumer<? super Object> action) {
    values.forEach(action);
  }

  @Override
  public int size() {
    return values.size();
  }

  @Override
  public double getDouble(int index) {
    return Convert.to(Double.class, values.get(index));
  }
}
