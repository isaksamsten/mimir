package org.briljantframework.mimir;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

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
  public boolean add(T t) {
    return inputs.add(t);
  }

  @Override
  public boolean addAll(Collection<? extends T> c) {
    return inputs.addAll(c);
  }

  @Override
  public Iterator<T> iterator() {
    return inputs.iterator();
  }

  @Override
  public int size() {
    return inputs.size();
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
