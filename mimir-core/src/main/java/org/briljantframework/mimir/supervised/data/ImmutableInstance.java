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

import static org.briljantframework.array.Arrays.unmodifiableArray;

import java.util.Arrays;

import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;

/**
 * Created by isak on 3/29/16.
 */
class ImmutableInstance implements Instance {

  private final Array<?> categoricalAttributes;
  private final DoubleArray numericalAttributes;

  ImmutableInstance(Object[] categoricalAttributes, double[] numericalAttributes) {
    this.categoricalAttributes = unmodifiableArray(
        Array.of(Arrays.copyOf(categoricalAttributes, categoricalAttributes.length)));
    this.numericalAttributes = unmodifiableArray(
        DoubleArray.of(Arrays.copyOf(numericalAttributes, numericalAttributes.length)));
  }

  ImmutableInstance(Array<?> categoricalAttributes, DoubleArray numericalAttributes) {
    this.categoricalAttributes = unmodifiableArray(categoricalAttributes.copy());
    this.numericalAttributes = unmodifiableArray(numericalAttributes.copy());
  }

  @Override
  public int numericalAttributes() {
    return numericalAttributes.size();
  }

  @Override
  public int categoricalAttributes() {
    return categoricalAttributes.size();
  }

//  @Override
//  public boolean isNA(int index) {
//    if (isNumericalAttribute(index)) {
//      return Is.NA(getNumericalAttribute(index));
//    } else {
//      return Is.NA(getCategoricalAttribute(index));
//    }
//  }

  @Override
  public double getNumericalAttribute(int index) {
    return numericalAttributes.get(index);
  }

  @Override
  public Object getCategoricalAttribute(int index) {
    return categoricalAttributes.get(index);
  }

  @Override
  public DoubleArray getNumericalAttributes() {
    if (hasNumericalAttributes())
      return numericalAttributes;
    else {
      throw new IllegalStateException();
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public Array<Object> getCategoricalAttributes() {
    if (hasCategoricalAttributes())
      return (Array<Object>) categoricalAttributes;
    else {
      throw new IllegalStateException();
    }
  }
}
