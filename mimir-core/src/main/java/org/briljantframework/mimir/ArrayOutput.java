package org.briljantframework.mimir;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * Created by isak on 3/10/16.
 */
public class ArrayOutput<T> implements Output<T> {

  private final ArrayList<T> values = new ArrayList<>();

  public ArrayOutput(Collection<? extends T> collection) {
    values.addAll(collection);
  }

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
  public <T1> T1[] toArray(T1[] a) {
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
  public T get(int i) {
    return values.get(i);
  }

  @Override
  public int size() {
    return values.size();
  }
}
