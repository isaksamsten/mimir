package org.briljantframework.mimir;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * Converts a {@link DataFrame} to an input, automatically setting the properties
 * 
 * <ul>
 * <li>{@link Dataset#FEATURE_SIZE}</li>
 * <li>{@link Dataset#FEATURE_TYPES}</li>
 * </ul>
 * 
 * @author Isak Karlsson
 */
class DataFrameInput extends AbstractInput<Instance> {
  private final DataFrame df;
  private final Properties properties;

  public DataFrameInput(DataFrame df) {
    Objects.requireNonNull(df, "DataFrame is required.");
    this.properties = createProperties(df);
    this.df = df;
  }

  /*
   * Creates a set of input properties based on the given dataframe
   */
  private static Properties createProperties(DataFrame df) {
    Properties properties = new Properties();
    List<Class<?>> types =
        df.getColumns().stream().map(v -> v.getType().getDataClass()).collect(Collectors.toList());
    properties.set(Dataset.FEATURE_TYPES, types);
    properties.set(Dataset.FEATURE_NAMES,
        df.getColumnIndex().keySet().stream().map(Object::toString).collect(Collectors.toList()));
    properties.set(Dataset.FEATURE_SIZE, df.columns());
    return properties;
  }

  @Override
  public Properties getProperties() {
    return properties;
  }

  @Override
  public int size() {
    return df.rows();
  }

  @Override
  public Instance get(int index) {
    return new RecordInstance(df.loc().getRecord(index));
  }

  private static class RecordInstance implements Instance {
    private final Vector record;

    public RecordInstance(Vector record) {
      this.record = record;
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
}
