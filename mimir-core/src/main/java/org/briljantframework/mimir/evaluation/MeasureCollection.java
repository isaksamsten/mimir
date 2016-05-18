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
package org.briljantframework.mimir.evaluation;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.series.DoubleSeries;
import org.briljantframework.data.series.Series;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class MeasureCollection {

  private final Map<Key, Series.Builder> measures;

  public MeasureCollection(MeasureCollection collection) {
    this.measures = new HashMap<>(collection.measures);
  }

  public MeasureCollection() {
    this.measures = new HashMap<>();
  }

  public synchronized void add(String measure, double value) {
    add(measure, MeasureSample.OUT_SAMPLE, value);
  }

  public synchronized void add(String measure, MeasureSample sample, double value) {
    measures.computeIfAbsent(new Key(measure, sample), (a) -> new DoubleSeries.Builder())
        .add(value);
  }

  public synchronized DataFrame toDataFrame() {
    DataFrame.Builder df = DataFrame.builder();
    for (Map.Entry<Key, Series.Builder> entry : measures.entrySet()) {
      if (entry.getKey().sample == MeasureSample.IN_SAMPLE) {
        continue;
      }
      // TODO: don't ignore in sample measures
      // TODO: ensure that all measures are of the same length
      // TODO: measures with both in-sample and out-sample will be overwritten
      df.set(entry.getKey().measure, entry.getValue());
    }
    return DataFrames.sortColumns(df.build(), Comparator.comparing(Object::toString));
  }

  private final static class Key {

    private final String measure;
    private final MeasureSample sample;

    private Key(String measure, MeasureSample sample) {
      this.measure = measure;
      this.sample = sample;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Key key = (Key) o;
      return !(measure != null ? !measure.equals(key.measure) : key.measure != null)
          && sample == key.sample;

    }

    @Override
    public int hashCode() {
      int result = measure != null ? measure.hashCode() : 0;
      result = 31 * result + (sample != null ? sample.hashCode() : 0);
      return result;
    }
  }
}
