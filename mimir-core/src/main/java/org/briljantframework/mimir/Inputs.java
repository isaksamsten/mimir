package org.briljantframework.mimir;



import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * Created by isak on 3/11/16.
 */
public final class Inputs {

  private Inputs() {}

  public static int features(Input<? extends Vector> tabularInput) {
    int size = tabularInput.get(0).size();
    for (int i = 1; i < tabularInput.size(); i++) {
      if (size != tabularInput.get(i).size()) {
        throw new IllegalArgumentException("Not a tabular input");
      }
    }
    return size;
  }

  public static DoubleArray toDoubleArray(Input<? extends Vector> x) {
    int n = x.size();
    int m = features(x);
    DoubleArray out = DoubleArray.zeros(n, m);
    for (int i = 0; i < n; i++) {
      Vector v = x.get(i);
      for (int j = 0; j < m; j++) {
        out.set(i, j, v.loc().getAsDouble(j));
      }
    }
    return out;
  }
}
