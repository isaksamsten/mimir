package org.briljantframework.mimir.data;

import org.briljantframework.data.vector.Vector;

import java.util.Objects;

/**
 * @author Isak Karlsson
 */
class VectorInstance implements Instance {
  private final Vector record;

  public VectorInstance(Vector record) {
    this.record = Objects.requireNonNull(record);
  }

  @Override
  public int size() {
    return record.size();
  }

  @Override
  public int getAsInt(int index) {
    return record.loc().getAsInt(index);
  }

  @Override
  public double getAsDouble(int index) {
    return record.loc().getAsDouble(index);
  }

  @Override
  public <T> T get(Class<T> cls, int index) {
    return record.loc().get(cls, index);
  }

  @Override
  public Object get(int index) {
    return record.loc().get(index);
  }
}
