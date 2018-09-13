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
package org.briljantframework.mimir.classification.tree;

import org.briljantframework.data.Is;
import org.briljantframework.mimir.supervised.data.Instance;

/**
 * Created by isak on 2017-02-28.
 */
class CategoricalNodeTest implements TreeNodeTest<Instance> {
  private final int ax;
  private final Object categoricalAttribute;

  public CategoricalNodeTest(int ax, Object categoricalAttribute) {
    this.ax = ax;
    this.categoricalAttribute = categoricalAttribute;
  }

  @Override
  public Direction test(Instance ex) {
    Object categoricalAttribute = ex.getCategoricalAttribute(ax);
    if (Is.NA(categoricalAttribute)) { // do null check
      return Direction.MISSING;
    } else {
      return categoricalAttribute.equals(this.categoricalAttribute) ? Direction.LEFT
          : Direction.RIGHT;
    }
  }
}
