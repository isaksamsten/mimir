package org.briljantframework.mimir;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;

/**
 * @author Isak Karlsson
 */
public class ArrayInput<T> extends AbstractInput<T> {

  private final Properties properties;
  private final ArrayList<T> inputs = new ArrayList<>();

  public ArrayInput() {
    this.properties = new Properties();
  }

  public ArrayInput(Collection<? extends T> collection) {
    this();
    inputs.addAll(collection);
  }

  public ArrayInput(Properties properties) {
    this.properties = new Properties(properties);
  }

  public ArrayInput(Input<? extends T> input) {
    this(input.getProperties());
    addAll(input);
  }

  @Override
  public boolean addAll(Collection<? extends T> c) {
    return inputs.addAll(c);
  }

  public boolean addAll(int index, Collection<? extends T> c) {
    return inputs.addAll(index, c);
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    return inputs.removeAll(c);
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    return inputs.retainAll(c);
  }

  @Override
  public Iterator<T> iterator() {
    return inputs.iterator();
  }

  @Override
  public void forEach(Consumer<? super T> action) {
    inputs.forEach(action);
  }

  @Override
  public Spliterator<T> spliterator() {
    return inputs.spliterator();
  }

  @Override
  public boolean removeIf(Predicate<? super T> filter) {
    return inputs.removeIf(filter);
  }

  public void replaceAll(UnaryOperator<T> operator) {
    inputs.replaceAll(operator);
  }

  @Override
  public int size() {
    return inputs.size();
  }

  @Override
  public boolean isEmpty() {
    return inputs.isEmpty();
  }

  @Override
  public boolean contains(Object o) {
    return inputs.contains(o);
  }

  public int indexOf(Object o) {
    return inputs.indexOf(o);
  }

  public int lastIndexOf(Object o) {
    return inputs.lastIndexOf(o);
  }

  @Override
  public Object[] toArray() {
    return inputs.toArray();
  }

  @Override
  public <T1> T1[] toArray(T1[] a) {
    return inputs.toArray(a);
  }

  public T set(int index, T element) {
    return inputs.set(index, element);
  }

  public T remove(int index) {
    return inputs.remove(index);
  }

  @Override
  public boolean add(T t) {
    return inputs.add(t);
  }

  public void add(int index, T element) {
    inputs.add(index, element);
  }

  @Override
  public boolean remove(Object o) {
    return inputs.remove(o);
  }

  @Override
  public Properties getProperties() {
    return properties;
  }

  @Override
  public T get(int index) {
    return inputs.get(index);
  }
}
