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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.briljantframework.Check;
import org.briljantframework.data.Is;
import org.briljantframework.data.statistics.FastStatistics;
import org.briljantframework.mimir.data.*;

/**
 * Z normalization is also known as "Normalization to Zero Mean and Unit of Energy" first mentioned
 * in Goldin & Kanellakis. It ensures that all elements of the input vector are transformed into the
 * output vector whose mean is approximately 0 while the standard deviation are in a range close to
 * 1.
 *
 * @author Isak Karlsson
 */
public class ZNormalizer<T extends Instance> implements Transformation<T, Instance> {

  @Override
  public Transformer<T, Instance> fit(Input<? extends T> x) {
    PropertyPreconditions.checkProperties(
        Arrays.asList(Dataset.FEATURE_SIZE, Dataset.FEATURE_TYPES), x.getProperties());
    List<FastStatistics> stats = new ArrayList<>();
    int columns = x.getProperty(Dataset.FEATURE_SIZE);
    List<Class<?>> featureType = x.getProperty(Dataset.FEATURE_TYPES);
    for (int i = 0; i < columns; i++) {
      stats.add(new FastStatistics());
    }

    for (T instance : x) {
      Check.state(instance.size() == columns);
      for (int i = 0; i < instance.size(); i++) {
        if (Number.class.isAssignableFrom(featureType.get(i))) {
          double value = instance.getDouble(i);
          if (!Is.NA(value)) {
            stats.get(i).addValue(value);
          }
        }
      }
    }


    List<StatisticalSummary> summaries =
        stats.stream().map(FastStatistics::getSummary).collect(Collectors.toList());
    return new ZNormalizerTransformer<>(featureType, summaries);
  }

  @Override
  public String toString() {
    return "ZNormalizer";
  }

  private static class ZNormalizerTransformer<T extends Instance>
      implements Transformer<T, Instance> {

    private final List<StatisticalSummary> summaries;
    private final List<Class<?>> featureTypes;

    ZNormalizerTransformer(List<Class<?>> featureTypes,
        List<StatisticalSummary> statisticalSummaries) {
      this.featureTypes = featureTypes;
      this.summaries = statisticalSummaries;
    }

    @Override
    public Instance transform(T x) {
      Check.state(x.size() == summaries.size());
      ArrayInstance o = new ArrayInstance();
      for (int i = 0; i < x.size(); i++) {
        double value = x.getDouble(i);
        if (Is.NA(value) || !Number.class.isAssignableFrom(featureTypes.get(i))) {
          o.add(x.get(i));
        } else {
          StatisticalSummary summary = summaries.get(i);
          o.add((value - summary.getMean()) / summary.getStandardDeviation());
        }
      }

      return o;
    }

    @Override
    public String toString() {
      return "ZNormalizerTransformer{" + "summaries=" + summaries + '}';
    }
  }
}
