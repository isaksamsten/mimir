package org.briljantframework.mimir.jfree;

import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.jfree.data.general.AbstractDataset;
import org.jfree.data.statistics.StatisticalCategoryDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class StatisticalDataFrameDataset extends AbstractDataset implements
    StatisticalCategoryDataset {

  private final Vector stats;

  public StatisticalDataFrameDataset(DataFrame dataFrame) {
    this.stats = dataFrame.reduce(Vector::statisticalSummary);
  }

  @Override
  public Number getMeanValue(int row, int column) {
    return stats.get(StatisticalSummary.class, stats.getIndex().get(column)).getMean();
  }

  @Override
  public Number getMeanValue(Comparable rowKey, Comparable columnKey) {
    return stats.get(StatisticalSummary.class, columnKey).getMean();
  }

  @Override
  public Number getStdDevValue(int row, int column) {
    return stats.get(StatisticalSummary.class, stats.getIndex().get(column)).getStandardDeviation();
  }

  @Override
  public Number getStdDevValue(Comparable rowKey, Comparable columnKey) {
    return stats.get(StatisticalSummary.class, columnKey).getStandardDeviation();
  }

  @Override
  public Comparable getRowKey(int row) {
    return 0;
  }

  @Override
  public int getRowIndex(Comparable key) {
    return 0;
  }

  @Override
  public List getRowKeys() {
    return Collections.singletonList(0);
  }

  @Override
  public Comparable getColumnKey(int column) {
    return (Comparable) stats.getIndex().get(column);
  }

  @Override
  public int getColumnIndex(Comparable key) {
    return stats.getIndex().getLocation(key);
  }

  @Override
  public List getColumnKeys() {
    return stats.getIndex();
  }

  @Override
  public Number getValue(int row, int column) {
    return getMeanValue(row, column);
  }

  @Override
  public Number getValue(Comparable rowKey, Comparable columnKey) {
    return getMeanValue(rowKey, columnKey);
  }

  @Override
  public int getRowCount() {
    return 1;
  }

  @Override
  public int getColumnCount() {
    return stats.size();
  }
}
