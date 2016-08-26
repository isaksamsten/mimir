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
package org.briljantframework.mimir.data.transform;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.*;

/**
 * @author Isak Karlsson
 */
public class MeanImputer<T extends Instance> implements Transformation<T, Instance> {

  @Override
  public Transformer<T, Instance> fit(Input<? extends T> x) {
    PropertyPreconditions.checkProperties(
        Arrays.asList(Dataset.FEATURE_SIZE, Dataset.FEATURE_TYPES), x.getProperties());
    int columns = x.getProperty(Dataset.FEATURE_SIZE);
    List<Class<?>> featureTypes = x.getProperty(Dataset.FEATURE_TYPES);
    DoubleArray means = DoubleArray.zeros(columns);
    for (int i = 0; i < x.size(); i++) {
      T instance = x.get(i);
      if (instance.size() != columns) {
        throw new IllegalStateException("Illegal sequence size.");
      }
      for (int j = 0; j < instance.size(); j++) {
        if (Number.class.isAssignableFrom(featureTypes.get(j))) {
          double value = instance.getDouble(j);
          if (!Is.NA(value)) {
            org.briljantframework.array.Arrays.plusAssign(means, DoubleArray.of(value / x.size()));
          }
        }
      }
    }

    return new MeanImputeTransformer<>(means);
  }

  @Override
  public String toString() {
    return "MeanImputer{}";
  }

  private static class MeanImputeTransformer<T extends Instance>
      implements Transformer<T, Instance> {

    private final DoubleArray means;

    MeanImputeTransformer(DoubleArray means) {
      this.means = means;
    }

    @Override
    public String toString() {
      return "MeanImputTransformer{" + "means=" + means + '}';
    }

    @Override
    public Input<Instance> transform(Input<? extends T> x) {
      PropertyPreconditions.checkProperties(
          Arrays.asList(Dataset.FEATURE_TYPES, Dataset.FEATURE_SIZE), x.getProperties());
      int columns = x.getProperty(Dataset.FEATURE_SIZE);
      Check.state(columns == means.size(), "illegal feature size");

      ArrayInput<Instance> o = x.stream().map(this::transform)
          .collect(Collectors.toCollection(() -> new ArrayInput<>(x.getProperties())));
      return Inputs.unmodifiableInput(o);
    }

    @Override
    public Instance transform(T x) {
      Check.state(x.size() == means.size(), "illegal feature size");
      Series.Builder vector = Series.Builder.of(Object.class);
      for (int i = 0; i < x.size(); i++) {
        if (Number.class.isAssignableFrom(x.getType(i))) {
          double value = x.getDouble(i);
          if (Is.NA(value)) {
            vector.set(i, means.get(i));
          } else {
            vector.set(i, value);
          }
        }
      }
      return Instance.of(vector);
    }
  }
}
