package org.briljantframework.mimir.jfree;

import org.briljantframework.array.Array;
import org.jfree.data.xy.XYZDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ArrayXYZDataset extends ArrayXYDataset implements XYZDataset {

  private final Array<? extends Number> z;

  public ArrayXYZDataset(Array<? extends Number> x, Array<? extends Number> y,
      Array<? extends Number> z) {
    super(x, y);
    this.z = z;
  }

  @Override
  public Number getZ(int series, int item) {
    return z.get(item, series);
  }

  @Override
  public double getZValue(int series, int item) {
    return getZ(series, item).doubleValue();
  }
}
