package org.briljantframework.mimir;

import org.briljantframework.data.vector.Convert;

import java.util.Collection;

/**
 * An output represents a collection of indexed output variables.
 * 
 * @author Isak Karlsson
 */
public interface Output<T> extends Collection<T> {

  /**
   * Return the output variable at the specified index.
   * 
   * @param index the
   * @return the output at the specified position
   */
  T get(int index);

  default <E> E get(Class<? extends E> cls, int index) {
    return Convert.to(cls, index);
  }
}
