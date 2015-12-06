package org.briljantframework.mimir.jfree;

import org.briljantframework.array.Array;
import org.jfree.data.xy.IntervalXYDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ArrayIntervalXYDataset extends ArrayXYDataset implements IntervalXYDataset {

  private final Array<? extends Number> error;

  public ArrayIntervalXYDataset(Array<? extends Number> x, Array<? extends Number> y,
      Array<? extends Number> error) {
    super(x, y);
    this.error = error;
  }

  @Override
  public Number getStartX(int series, int item) {
    return getXValue(series, item);
  }

  @Override
  public double getStartXValue(int series, int item) {
    return getStartX(series, item).doubleValue();
  }

  @Override
  public Number getEndX(int series, int item) {
    return getXValue(series, item);
  }

  @Override
  public double getEndXValue(int series, int item) {
    return getEndX(series, item).doubleValue();
  }

  @Override
  public Number getStartY(int series, int item) {
    return getStartYValue(series, item);
  }

  @Override
  public double getStartYValue(int series, int item) {
    return getYValue(series, item) - error.get(item, series).doubleValue();
  }

  @Override
  public Number getEndY(int series, int item) {
    return getEndYValue(series, item);
  }

  @Override
  public double getEndYValue(int series, int item) {
    return getYValue(series, item) + error.get(item, series).doubleValue();
  }
}
