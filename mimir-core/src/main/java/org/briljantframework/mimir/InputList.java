package org.briljantframework.mimir;

import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by isak on 3/10/16.
 */
public class InputList<T> extends AbstractCollection<T> implements Input<T> {
  private ArrayList<T> inputs = new ArrayList<>();
  private final InputProperties properties = new InputProperties();

  @Override
  public Iterator<T> iterator() {
    return inputs.iterator();
  }

  @Override
  public int size() {
    return inputs.size();
  }

  @Override
  public InputProperties getProperties() {
    return properties;
  }

  @Override
  public T get(int row) {
    return inputs.get(row);
  }
}
