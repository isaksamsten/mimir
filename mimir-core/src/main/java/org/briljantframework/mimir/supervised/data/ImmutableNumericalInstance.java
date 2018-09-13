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
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;

import java.util.stream.Collectors;

/**
 * Created by isak on 2017-02-27.
 */
class ImmutableNumericalInstance implements Instance {
  private final DoubleArray numericalAttributes;

  ImmutableNumericalInstance(DoubleArray numericalAttributes) {
    this.numericalAttributes = Arrays.unmodifiableArray(numericalAttributes.copy());
  }

  @Override
  public int numericalAttributes() {
    return numericalAttributes.size();
  }

  @Override
  public int categoricalAttributes() {
    return 0;
  }

//  @Override
//  public boolean isNA(int index) {
//    Check.index(index, numericalAttributes());
//    return Is.NA(getNumericalAttribute(index));
//  }

  @Override
  public double getNumericalAttribute(int index) {
    return numericalAttributes.get(index);
  }

  @Override
  public Object getCategoricalAttribute(int index) {
    throw new IndexOutOfBoundsException();
  }

  @Override
  public DoubleArray getNumericalAttributes() {
    return numericalAttributes;
  }

  @Override
  public Array<Object> getCategoricalAttributes() {
    throw new IllegalStateException();
  }

  @Override
  public String toString() {
    return numericalAttributes.stream().map(d -> String.format("%.2f", d)).collect(Collectors.joining(", "));
  }

}
