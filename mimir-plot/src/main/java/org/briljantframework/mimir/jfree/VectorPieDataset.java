package org.briljantframework.mimir.jfree;

import java.util.List;

import org.briljantframework.data.vector.Vector;
import org.jfree.data.general.AbstractDataset;
import org.jfree.data.general.PieDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class VectorPieDataset extends AbstractDataset implements PieDataset {

  private final Vector vector;

  public VectorPieDataset(Vector vector) {
    this.vector = vector;
  }

  @Override
  public Comparable getKey(int index) {
    return (Comparable) vector.getIndex().get(index);
  }

  @Override
  public int getIndex(Comparable key) {
    return vector.getIndex().getLocation(key);
  }

  @Override
  public List getKeys() {
    return vector.getIndex();
  }

  @Override
  public Number getValue(Comparable key) {
    return vector.get(Number.class, key);
  }

  @Override
  public int getItemCount() {
    return vector.size();
  }

  @Override
  public Number getValue(int index) {
    return vector.get(Number.class, vector.getIndex().get(index));
  }
}
