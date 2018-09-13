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

import org.briljantframework.Check;

/**
 * A mutable input collection.
 * 
 * @author Isak Karlsson
 */
public class MutableInput<T> extends Input<T> implements RandomAccess {

  private final ArrayList<T> elements = new ArrayList<>();
  private final Schema<T> schema;

  public MutableInput(Schema<T> schema) {
    this.schema = schema;
  }

  public MutableInput(Input<T> input) {
    this(input.getSchema());
    elements.addAll(input);
  }

  @Override
  public boolean add(T t) {
    Check.argument(getSchema().isValid(t), ILLEGAL_SCHEMA);
    return elements.add(t);
  }

  public void add(int index, T element) {
    Check.argument(getSchema().isValid(element), ILLEGAL_SCHEMA);
    elements.add(index, element);
  }

  @Override
  public boolean addAll(Collection<? extends T> c) {
    Check.all(c).argument(getSchema()::isValid, ILLEGAL_SCHEMA);
    return elements.addAll(c);
  }

  public boolean addAll(int index, Collection<? extends T> c) {
    Check.all(c).argument(getSchema()::isValid, ILLEGAL_SCHEMA);
    return elements.addAll(index, c);
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    return elements.removeAll(c);
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    return elements.retainAll(c);
  }

  @Override
  public Iterator<T> iterator() {
    return elements.iterator();
  }

  @Override
  public Schema<T> getSchema() {
    return schema;
  }

  @Override
  public void forEach(Consumer<? super T> action) {
    elements.forEach(action);
  }

  @Override
  public Spliterator<T> spliterator() {
    return elements.spliterator();
  }

  @Override
  public boolean removeIf(Predicate<? super T> filter) {
    return elements.removeIf(filter);
  }

  public void replaceAll(UnaryOperator<T> operator) {
    UnaryOperator<T> safeOperator = (e) -> {
      T element = operator.apply(e);
      if (getSchema().isValid(element)) {
        return element;
      } else {
        throw new IllegalStateException(ILLEGAL_SCHEMA);
      }
    };
    elements.replaceAll(safeOperator);
  }

  @Override
  public int size() {
    return elements.size();
  }

  @Override
  public boolean isEmpty() {
    return elements.isEmpty();
  }

  @Override
  public boolean contains(Object o) {
    return elements.contains(o);
  }

  public int indexOf(Object o) {
    return elements.indexOf(o);
  }

  public int lastIndexOf(Object o) {
    return elements.lastIndexOf(o);
  }

  @Override
  public Object[] toArray() {
    return elements.toArray();
  }

  @Override
  public <T1> T1[] toArray(T1[] a) {
    return elements.toArray(a);
  }

  public T set(int index, T element) {
    Check.argument(getSchema().isValid(element), ILLEGAL_SCHEMA);
    return elements.set(index, element);
  }

  public T remove(int index) {
    return elements.remove(index);
  }

  @Override
  public boolean remove(Object o) {
    return elements.remove(o);
  }

  @Override
  public T get(int index) {
    return elements.get(index);
  }
}
