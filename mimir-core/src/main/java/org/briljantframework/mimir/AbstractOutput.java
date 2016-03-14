package org.briljantframework.mimir;

import java.util.AbstractCollection;
import java.util.Iterator;

/**
 * @author Isak Karlsson
 */
public abstract class AbstractOutput<T> extends AbstractCollection<T> implements Output<T> {
  @Override
  public Iterator<T> iterator() {
    return new Iterator<T>() {
      private int current = 0;

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
