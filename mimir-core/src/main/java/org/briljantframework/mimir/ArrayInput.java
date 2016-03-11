package org.briljantframework.mimir;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * Created by isak on 3/10/16.
 */
public class ArrayInput<T> extends AbstractInput<T> {

  private final ArrayList<T> inputs = new ArrayList<>();


  public ArrayInput() {}

  public ArrayInput(Collection<? extends T> collection) {
    inputs.addAll(collection);
  }

  public ArrayInput(InputProperties properties) {
    super(new InputProperties(properties));
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
  public T get(int row) {
    return inputs.get(row);
  }
}
