package org.briljantframework.mimir;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * Created by isak on 3/11/16.
 */
public class DataFrameInput extends AbstractInput<Vector> {
  private final DataFrame df;

  public DataFrameInput(DataFrame df) {
    super(createProperties(df));
    this.df = df;

  }

  private static InputProperties createProperties(DataFrame df) {
    InputProperties properties = new InputProperties();
    List<Class> types =
        df.getColumns().stream().map(v -> v.getType().getDataClass()).collect(Collectors.toList());
    properties.set(Dataset.TYPES, types);
    properties.set(Dataset.FEATURES, df.columns());
    return properties;
  }

  @Override
  public Iterator<Vector> iterator() {
    return df.getRecords().iterator();
  }

  @Override
  public int size() {
    return df.rows();
  }

  @Override
  public Vector get(int row) {
    return df.loc().getRecord(row);
  }
}
