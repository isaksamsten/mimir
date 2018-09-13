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

import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.data.series.Series;

/**
 * @author Isak Karlsson
 */
class SeriesInstance implements Instance {
  private final Series record;
  private final int[] categoricalAttributes;
  private final int[] numericalAttributes;

  SeriesInstance(Series record, int[] categoricalAttributeMap, int[] numericalAttributeMap) {
    this.record = Objects.requireNonNull(record);
    this.numericalAttributes = numericalAttributeMap;
    this.categoricalAttributes = categoricalAttributeMap;
  }

  @Override
  public int numericalAttributes() {
    return numericalAttributes.length;
  }

  @Override
  public int categoricalAttributes() {
    return categoricalAttributes.length;
  }

  @Override
  public double getNumericalAttribute(int index) {
    Check.index(index, numericalAttributes());
    return record.values().getDouble(numericalAttributes[index]);
  }

  @Override
  public Object getCategoricalAttribute(int index) {
    Check.index(index, categoricalAttributes());
    return record.values().get(categoricalAttributes[index]);
  }
}
