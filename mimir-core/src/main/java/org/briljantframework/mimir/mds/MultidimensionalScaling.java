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
package org.briljantframework.mimir.mds;


import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.Matrices;
import org.briljantframework.mimir.data.transform.Transformer;

/**
 * Classical multidimensional scaling (MDS) of a data matrix. Also known as principal coordinates
 * analysis.
 *
 * <p/>
 * Multidimensional scaling takes a set of dissimilarities and returns a set of points such that the
 * distances between the points are approximately equal to the dissimilarities.
 *
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class MultidimensionalScaling implements Transformer<DoubleArray, DoubleArray> {

  private final int components;

  public MultidimensionalScaling(int k) {
    this.components = k;
  }

  public DoubleArray transform(DoubleArray proximity) {
    Check.argument(proximity.isMatrix() && proximity.isSquare(), "non-square proximity matrix");
    Check.argument(components > 0 && components <= proximity.size(0), "illegal component size");
    int m = proximity.size(0);

    DoubleArray a = DoubleArray.zeros(m, m);
    DoubleArray b = DoubleArray.zeros(m, m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < i; j++) {
        a.set(i, j, -0.5 * Math.sqrt(proximity.get(i, j)));
        a.set(j, i, a.get(i, j));
      }
    }

    DoubleArray means = Arrays.mean(0, a);
    double mu = Arrays.mean(means);
    for (int i = 0; i < m; i++) {
      double meanI = means.get(i);
      for (int j = 0; j < i; j++) {
        b.set(i, j, a.get(i, j) - meanI - means.get(j) + mu);
        b.set(j, i, b.get(i, j));
      }
    }
    // TODO: 06/12/15 Use the lapack routine to improve performance
    EigenDecomposition decomposition = new EigenDecomposition(Matrices.asRealMatrix(b));
    DoubleArray realEigenvalues = DoubleArray.of(decomposition.getRealEigenvalues());
    DoubleArray eigenvectors = Matrices.toArray(decomposition.getV());

    DoubleArray coordinates = DoubleArray.zeros(m, components);
    for (int j = 0; j < components; j++) {
      double eig = realEigenvalues.get(j);
      if (eig < 0) {
        throw new NotStrictlyPositiveException(eig);
      }
      double scale = Math.sqrt(eig);
      for (int i = 0; i < m; i++) {
        coordinates.set(i, j, eigenvectors.get(i, j) * scale);
      }
    }
    return coordinates;
  }
}
