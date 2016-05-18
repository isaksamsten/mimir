/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.briljantframework.mimir.data;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import org.briljantframework.data.dataframe.DataFrame;

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
  private final TypeMap properties;

  public DataFrameInput(DataFrame df) {
    Objects.requireNonNull(df, "DataFrame is required.");
    this.properties = createProperties(df);
    this.df = df;
  }

  /*
   * Creates a set of input properties based on the given dataframe
   */
  private static TypeMap createProperties(DataFrame df) {
    TypeMap properties = new TypeMap();
    List<Class<?>> types =
        df.columns().stream().map(v -> v.getType().getDataClass()).collect(Collectors.toList());
    properties.set(Dataset.FEATURE_TYPES, types);
    properties.set(Dataset.FEATURE_NAMES,
        df.getColumnIndex().keySet().stream().map(Object::toString).collect(Collectors.toList()));
    properties.set(Dataset.FEATURE_SIZE, df.size(1));
    return properties;
  }

  @Override
  public TypeMap getProperties() {
    return properties;
  }

  @Override
  public int size() {
    return df.size(0);
  }

  @Override
  public Instance get(int index) {
    return new SeriesInstance(df.loc().getRow(index));
  }

}
