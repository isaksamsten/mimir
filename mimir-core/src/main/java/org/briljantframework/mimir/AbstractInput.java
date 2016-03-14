package org.briljantframework.mimir;

import java.util.AbstractCollection;
import java.util.Iterator;

/**
 * @author Isak Karlsson
 */
public abstract class AbstractInput<T> extends AbstractCollection<T> implements Input<T> {

  @Override
  public Iterator<T> iterator() {
    return new Iterator<T>() {
      public int current = 0;

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
}
