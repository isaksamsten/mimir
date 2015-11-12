/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.distance;

import java.util.function.DoubleSupplier;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * Created by isak on 24/03/15.
 */
public class EditDistance implements Distance {

  @Override
  public double compute(double a, double b) {
    return 0;
  }

  @Override
  public double compute(Vector a, Vector b) {
    if (a.size() < b.size()) {
      return compute(b, a);
    }

    if (b.size() == 0) {
      return a.size();
    }

    DoubleSupplier iter = new DoubleSupplier() {
      private int i = 0;

      @Override
      public double getAsDouble() {
        return i++;
      }
    };

    DoubleArray previousRow = Arrays.newDoubleArray(b.size() + 1);
    previousRow.assign(iter);
    for (int i = 0; i < a.size(); i++) {
      DoubleArray currentRow = Arrays.newDoubleArray(b.size() + 1);
      currentRow.set(0, i + 1);
      for (int j = 0; j < b.size(); j++) {
        double insert = previousRow.get(j + 1) + 1;
        double delete = currentRow.get(j) + 1;
        double subs = previousRow.get(j) + (!a.loc().equals(i, b, j) ? 1 : 0);
        currentRow.set(j + 1, Math.min(insert, Math.min(delete, subs)));
      }
      previousRow = currentRow;
    }

    return previousRow.get(previousRow.size() - 1);
  }

  @Override
  public double max() {
    return 0;
  }

  @Override
  public double min() {
    return 0;
  }
}
