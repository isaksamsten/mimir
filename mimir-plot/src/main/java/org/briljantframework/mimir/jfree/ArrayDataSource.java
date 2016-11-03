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
package org.briljantframework.mimir.jfree;

import org.briljantframework.array.Array;

import de.erichseifert.gral.data.AbstractDataSource;
import org.briljantframework.array.ComplexArray;
import org.jfree.data.ComparableObjectItem;

/**
 * Created by isak on 2016-09-27.
 */
public class ArrayDataSource extends AbstractDataSource {

  private final Array<? extends Number> array;

  public ArrayDataSource(Array<? extends Number> array) {
    super(numberArray(array));
    this.array = array;
  }

  private static Class<? extends Comparable<?>>[] numberArray(Array<? extends Number> array) {
    Class[] c = new Class[array.columns()];
    for (int i = 0; i < array.columns(); i++) {
      c[i] = Number.class;
    }
    return c;
  }

  @Override
  public Comparable<?> get(int i, int i1) {
    return (Comparable<?>) array.get(i, i1);
  }

  @Override
  public int getRowCount() {
    return array.rows();
  }
}
