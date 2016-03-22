/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.briljantframework.mimir.data;

import org.briljantframework.DoubleSequence;
import org.briljantframework.data.vector.Convert;

/**
 * An m-feature vector of heterogeneous values. Generally, an input collection of instances should
 * report the {@link Dataset#FEATURE_SIZE} and {@link Dataset#FEATURE_TYPES} properties, where the
 * former reports the number of features in a dataset and the latter the types of values in the
 * dataset.
 * 
 * <p/>
 * To convert a {@link org.briljantframework.data.dataframe.DataFrame data frame} to an input of
 * instances, use the {@link DataFrameInput} class which automatically reports both properties.
 * 
 * @author Isak Karlsson
 */
public interface Instance extends DoubleSequence {

  /**
   * Returns the size of the instance (i.e., the number of features)
   * 
   * @return the size of the instance
   */
  @Override
  int size();

  /**
   * Returns the value of the i:th feature as an object.
   *
   * @param index the index
   * @return the value as an object
   */
  Object get(int index);

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
   * Returns the value of the i:th feature as an int.
   *
   * @param index the index
   * @return the value as an int
   */
  default int getAsInt(int index) {
    return get(Integer.class, index);
  }
}
