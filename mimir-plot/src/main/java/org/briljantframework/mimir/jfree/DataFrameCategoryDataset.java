package org.briljantframework.mimir.jfree;

import java.util.List;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.AbstractDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DataFrameCategoryDataset extends AbstractDataset implements CategoryDataset {

  private final DataFrame dataFrame;

  public DataFrameCategoryDataset(DataFrame dataFrame) {
    this.dataFrame = dataFrame;
  }

  @Override
  public Comparable getRowKey(int row) {
    return (Comparable) dataFrame.getIndex().get(row);
  }

  @Override
  public int getRowIndex(Comparable key) {
    return dataFrame.getIndex().getLocation(key);
  }

  @Override
  public List getRowKeys() {
    return dataFrame.getIndex();
  }

  @Override
  public Comparable getColumnKey(int column) {
    return (Comparable) dataFrame.getColumnIndex().get(column);
  }

  @Override
  public int getColumnIndex(Comparable key) {
    return dataFrame.getColumnIndex().getLocation(key);
  }

  @Override
  public List getColumnKeys() {
    return dataFrame.getColumnIndex();
  }

  @Override
  public Number getValue(Comparable rowKey, Comparable columnKey) {
    Number value = dataFrame.get(Number.class, rowKey, columnKey);
    if (Is.NA(value)) {
      return null;
    }
    return value;
  }

  @Override
  public int getRowCount() {
    return dataFrame.rows();
  }

  @Override
  public int getColumnCount() {
    return dataFrame.columns();
  }

  @Override
  public Number getValue(int row, int column) {
    Number value =
        dataFrame.get(Number.class, dataFrame.getIndex().get(row),
            dataFrame.getColumnIndex().get(column));
    if (Is.NA(value)) {
      return null;
    }
    return value;
  }
}
