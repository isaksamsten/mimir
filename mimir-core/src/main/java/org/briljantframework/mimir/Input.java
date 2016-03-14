package org.briljantframework.mimir;

import java.util.Collection;

/**
 * Input variables
 */
public interface Input<T> extends Collection<T> {

  /**
   * Returns a collection of properties for the given input.
   * 
   * @return the collection of properties for the input.
   */
  Properties getProperties();

  /**
   * Returns the specified property.
   * 
   * @param property the property
   * @return the value for the given property
   */
  default <E> E getProperty(Property<E> property) {
    return getProperties().get(property);
  }

  /**
   * Returns the input data at the specified index.
   * 
   * @param index the index
   * @return the input data at the specified index.
   */
  T get(int index);
}
