package org.briljantframework.mimir.jfree;

import java.util.List;

import org.briljantframework.array.Array;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.array.LongArray;
import org.jfree.data.xy.AbstractXYDataset;

/**
 * A data set for plotting a numerical array as the y-coordinate(s) (each column being a series) and
 * x being in the range 0..y.size(0)
 * 
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ArrayYDataset extends AbstractXYDataset {

  private final Array<? extends Number> y;
  private final List<? extends Comparable> seriesKeys;

  public ArrayYDataset(DoubleArray y) {
    this(y.boxed());
  }

  public ArrayYDataset(Array<? extends Number> y) {
    this(y, null);
  }

  public ArrayYDataset(Array<? extends Number> y, List<? extends Comparable> seriesKeys) {
    this.y = y;
    this.seriesKeys = seriesKeys;
  }

  public ArrayYDataset(IntArray y) {
    this(y.boxed());
  }

  public ArrayYDataset(LongArray y) {
    this(y.boxed());
  }

  public ArrayYDataset(BooleanArray y) {
    this(y.asInt().boxed());
  }

  @Override
  public int getSeriesCount() {
    return y.isVector() ? 1 : y.size(1);
  }

  @Override
  public Comparable getSeriesKey(int series) {
    return seriesKeys != null ? seriesKeys.get(series) : series;
  }

  @Override
  public int getItemCount(int series) {
    return y.size(0);
  }

  @Override
  public Number getX(int series, int item) {
    return item;
  }

  @Override
  public Number getY(int series, int item) {
    return y.isVector() ? y.get(item) : y.get(item, series);
  }
}
