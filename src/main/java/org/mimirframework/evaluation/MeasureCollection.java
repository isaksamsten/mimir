package org.mimirframework.evaluation;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.DoubleVector;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class MeasureCollection {

  private final Map<Key, Vector.Builder> measures;

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
    measures.computeIfAbsent(new Key(measure, sample), (a) -> new DoubleVector.Builder())
        .add(value);
  }

  public synchronized DataFrame toDataFrame() {
    DataFrame.Builder df = DataFrame.builder();
    for (Map.Entry<Key, Vector.Builder> entry : measures.entrySet()) {
      if (entry.getKey().sample == MeasureSample.IN_SAMPLE) {
        continue;
      }
      // TODO: don't ignore in sample measures
      // TODO: ensure that all measures are of the same length
      // TODO: measures with both in-sample and out-sample will be overwritten
      df.set(entry.getKey().measure, entry.getValue());
    }
    return df.build().sortColumns(Comparator.comparing(Object::toString));
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
