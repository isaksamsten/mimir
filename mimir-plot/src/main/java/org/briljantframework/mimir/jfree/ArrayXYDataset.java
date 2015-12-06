package org.briljantframework.mimir.jfree;

import java.util.List;

import org.briljantframework.array.Array;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ArrayXYDataset extends ArrayYDataset {

  private final Array<? extends Number> x;

  public ArrayXYDataset(Array<? extends Number> y, Array<? extends Number> x) {
    this(x, y, null);
  }

  public ArrayXYDataset(Array<? extends Number> x, Array<? extends Number> y,
      List<? extends Comparable> seriesKeys) {
    super(y, seriesKeys);
    this.x = x;
  }

  @Override
  public Number getX(int series, int item) {
    return x.get(item, series);
  }
}
