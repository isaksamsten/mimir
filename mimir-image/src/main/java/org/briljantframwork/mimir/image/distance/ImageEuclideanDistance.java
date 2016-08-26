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
package org.briljantframwork.mimir.image.distance;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.distance.Distance;

/**
 * Created by isak on 3/31/16.
 */
public class ImageEuclideanDistance implements Distance<DoubleArray> {

  private double sigma = 1;
  private double gamma = 0.5 / (sigma * sigma);

  @Override
  public double compute(DoubleArray x, DoubleArray y) {
    Check.argument(x.dims() == 2 && y.dims() == 2, "require square image");
    Check.argument(x.size(0) == y.size(0) && x.size(1) == y.size(1), "images must be of same size");
    int size = x.size();
    x = x.transpose();
    y = y.transpose();
    System.out.println(x);
    System.out.println(y);
    double sum = 0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        double spatial = spatial(i, i, j, j);
        double v1 = x.get(i) - y.get(i);
        double v2 = x.get(j) - y.get(j);
        double diff = v1 * v2;

        // double diff = Math.pow(x.get(i) - y.get(j), 2);
        double v = spatial * diff;
        sum += v;

      }
    }
    return Math.sqrt(sum);
  }

  private double spatial(int k, int l, int kp, int lp) {
    double ksub = Math.pow(k - kp, 2);
    double lsub = Math.pow(l - lp, 2);
    double dist = ksub + lsub;
    double exp = Math.exp(-gamma * dist);
    return exp;
    // return dist == 0 ? 1 : 0;
  }
}
