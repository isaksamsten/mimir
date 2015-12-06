package org.briljantframework.mimir.mds;


import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.Matrices;

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
public class MultidimensionalScaling {

  private final DoubleArray coordinates;

  public MultidimensionalScaling(DoubleArray proximity, int k) {
    Check.argument(proximity.isMatrix() && proximity.isSquare(), "non-square proximity matrix");
    Check.argument(k > 0 && k <= proximity.size(0), "illegal k");
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

    DoubleArray coordinates = DoubleArray.zeros(m, k);
    for (int j = 0; j < k; j++) {
      double eig = realEigenvalues.get(j);
      if (eig < 0) {
        throw new NotStrictlyPositiveException(eig);
      }
      double scale = Math.sqrt(eig);
      for (int i = 0; i < m; i++) {
        coordinates.set(i, j, eigenvectors.get(i, j) * scale);
      }
    }
    this.coordinates = coordinates;
  }

  public DoubleArray getCoordinates() {
    return coordinates.copy();
  }
}
