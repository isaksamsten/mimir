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
package org.briljantframework.mimir.supervised.data;

import java.util.Objects;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Schema;

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
public class DataFrameInput extends Input<Instance> {
  private final DataFrame df;
  private final MultidimensionalSchema schema;
  private final int[] categoricalAttributeMap;
  private final int[] numericalAttributeMap;

  public DataFrameInput(DataFrame df) {
    this.df = Objects.requireNonNull(df, "DataFrame is required.");
    int numericalAttributes = (int) df.getColumns().stream().map(Series::getType)
        .filter(t -> t.isAssignableTo(Number.class)).count();
    int categoricalAttributes = df.columns() - numericalAttributes;
    this.schema = new MultidimensionalSchema(numericalAttributes, categoricalAttributes);
    categoricalAttributeMap = new int[categoricalAttributes];
    numericalAttributeMap = new int[numericalAttributes];
    int n = 0, c = 0;
    for (int i = 0; i < df.columns(); i++) {
      if (df.loc().get(i).getType().isAssignableTo(Number.class)) {
        numericalAttributeMap[n] = i;
        this.schema.setAttributeName(n, df.getColumnIndex().get(i).toString());
        n += 1;
      } else {
        categoricalAttributeMap[c] = i;
        this.schema.setAttributeName(numericalAttributes - c,
            df.getColumnIndex().get(i).toString());
        c += 1;
      }
    }
  }

  @Override
  public int size() {
    return df.rows();
  }

  @Override
  public Instance get(int index) {
    return new SeriesInstance(df.loc().getRow(index), categoricalAttributeMap,
        numericalAttributeMap);
  }

  @Override
  public Schema<Instance> getSchema() {
    return schema;
  }
}
