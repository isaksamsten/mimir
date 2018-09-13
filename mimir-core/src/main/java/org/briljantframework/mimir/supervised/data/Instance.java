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
package org.briljantframework.mimir.supervised.data;

import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;

/**
 * An instance consists of {@code m} values out of which {@code c} are categorical and {@code n} are
 * numerical.
 *
 * <p/>
 * Categorical attributes are indexed from {@code 0 .. categoricalAttributes()} and numerical
 * attributes from {@code 0 .. numericalAttributes()}
 *
 * @author Isak Karlsson
 */
public interface Instance {

  static Instance of(Array<?> categoricalAttributes, DoubleArray numericalAttributes) {
    return new ImmutableInstance(categoricalAttributes, numericalAttributes);
  }

  static Instance of(Array<?> categoricalAttributes) {
    return new ImmutableCategoricalInstance(categoricalAttributes);
  }

  static Instance of(DoubleArray numericalAttributes) {
    return new ImmutableNumericalInstance(numericalAttributes);
  }


  /**
   * Return the number of numerical attributes of this instance
   *
   * @return the number of
   */
  int numericalAttributes();

  int categoricalAttributes();

  double getNumericalAttribute(int index);

  Object getCategoricalAttribute(int index);

  default boolean hasNumericalAttributes() {
    return numericalAttributes() > 0;
  }

  default boolean hasCategoricalAttributes() {
    return categoricalAttributes() > 0;
  }

  default DoubleArray getNumericalAttributes() {
    DoubleArray array = DoubleArray.zeros(numericalAttributes());
    for (int i = 0; i < numericalAttributes(); i++) {
      array.set(i, getNumericalAttribute(i));
    }
    return array;
  }

  default Array<Object> getCategoricalAttributes() {
    Array<Object> array = Array.empty(categoricalAttributes());
    for (int i = 0; i < categoricalAttributes(); i++) {
      array.set(i, getCategoricalAttribute(i));
    }
    return array;
  }

}
