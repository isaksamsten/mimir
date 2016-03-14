package org.briljantframework.mimir;

import org.briljantframework.data.vector.Convert;

/**
 * An m-feature vector of heterogeneous values. Generally, an input collection of instances should
 * report the {@link Dataset#FEATURES} and {@link Dataset#FEATURE_TYPES} properties, where the
 * former reports the number of features in a dataset and the latter the types of values in the
 * dataset.
 * 
 * <p/>
 * To convert a {@link org.briljantframework.data.dataframe.DataFrame data frame} to an input of
 * instances, use the {@link DataFrameInput} class which automatically reports both properties.
 * 
 * @author Isak Karlsson
 */
public interface Instance {

  /**
   * Returns the size of the instance (i.e., the number of features)
   * 
   * @return the size of the instance
   */
  int size();

  /**
   * Returns the value of the i:th feature as an int.
   * 
   * @param index the index
   * @return the value as an int
   */
  int getAsInt(int index);

  /**
   * Returns the value of the i:th feature as a double.
   * 
   * @param index the index
   * @return the value as a double
   */
  double getAsDouble(int index);

  /**
   * Returns the value of the i:th feature as an instance of the given class, or {@code NA} if the
   * conversion fails.
   * 
   * <p/>
   * The default implementation uses {@link Convert#to(Class, Object)} for conversion.
   * 
   * @param cls the class to return as
   * @param index the index
   * @return the value as an instance of the given class
   */
  default <T> T get(Class<T> cls, int index) {
    return Convert.to(cls, get(index));
  }

  /**
   * Returns the value of the i:th feature as an object.
   * 
   * @param index the index
   * @return the value as an object
   */
  Object get(int index);
}
