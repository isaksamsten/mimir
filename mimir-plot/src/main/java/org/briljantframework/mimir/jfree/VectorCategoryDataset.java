package org.briljantframework.mimir.jfree;

import java.util.Collections;
import java.util.List;

import org.briljantframework.data.Is;
import org.briljantframework.data.vector.Vector;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.AbstractDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class VectorCategoryDataset extends AbstractDataset implements CategoryDataset {

  private final Vector vector;

  public VectorCategoryDataset(Vector vector) {
    this.vector = vector;
  }

  @Override
  public Comparable getRowKey(int row) {
    return (Comparable) vector.getIndex().get(row);
  }

  @Override
  public int getRowIndex(Comparable key) {
    return vector.getIndex().getLocation(key);
  }

  @Override
  public List getRowKeys() {
    return vector.getIndex();
  }

  @Override
  public Comparable getColumnKey(int column) {
    return 0;
  }

  @Override
  public int getColumnIndex(Comparable key) {
    return 0;
  }

  @Override
  public List getColumnKeys() {
    return Collections.singletonList(0);
  }

  @Override
  public Number getValue(Comparable rowKey, Comparable columnKey) {
    return vector.get(Number.class, rowKey);
  }

  @Override
  public int getRowCount() {
    return vector.size();
  }

  @Override
  public int getColumnCount() {
    return 1;
  }

  @Override
  public Number getValue(int row, int column) {
    Number value = vector.get(Number.class, vector.getIndex().get(row));
    if (Is.NA(value)) {
      return null;
    }
    return value;
  }

}
