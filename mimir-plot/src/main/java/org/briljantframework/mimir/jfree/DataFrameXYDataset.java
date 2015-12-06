package org.briljantframework.mimir.jfree;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.jfree.data.xy.AbstractXYDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DataFrameXYDataset extends AbstractXYDataset {

  private final DataFrame dataFrame;

  public DataFrameXYDataset(DataFrame dataFrame) {
    this.dataFrame = dataFrame;
  }

  @Override
  public int getSeriesCount() {
    return dataFrame.columns();
  }

  @Override
  public Comparable getSeriesKey(int series) {
    return (Comparable) dataFrame.getColumnIndex().get(series);
  }

  @Override
  public int getItemCount(int series) {
    return dataFrame.rows();
  }

  @Override
  public Number getX(int series, int item) {
    return item;
  }

  @Override
  public Number getY(int series, int item) {
    Object r = dataFrame.getIndex().get(item);
    Object c = dataFrame.getColumnIndex().get(series);
    Number value = dataFrame.get(Number.class, r, c);
    if (Is.NA(value)) {
      return null;
    }
    return value;
  }

}
