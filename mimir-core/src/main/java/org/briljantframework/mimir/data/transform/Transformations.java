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
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.Na;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.briljantframework.data.statistics.FastStatistics;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.distance.Distance;

/**
 * Created by isak on 2016-09-30.
 */
public final class Transformations {

  private static final Normalizer<?> Z_NORMALIZER = new Normalizer<>();
  private static final MinMaxNormalizer<?> MIN_MAX_NORMALIZER = new MinMaxNormalizer<>();
  private static final MeanImputer<?> MEAN_IMPUTER = new MeanImputer<>();

  private Transformations() {}

  @SuppressWarnings("unchecked")
  public static <T extends Instance> InputTransformation<T, Instance> meanImputer() {
    return (InputTransformation<T, Instance>) MEAN_IMPUTER;
  }

  /**
   * Class to fit a min max normalizer to a data frame. Calculate, for each column {@code j}, the
   * min <i>min</i><minus>j</minus> and max <i>max</i><minus>j</minus>. Then, for each value x
   * <minus>i,j</minus> is given by (x<minus>i,j</minus>-min<minus>j</minus>)/(max<minus>j</minus> -
   * min<minus>j</minus>). This normalizes the data frame in the range {@code [0, 1]} (under the
   * assumption that min and max are representative for the transformed dataframe).
   *
   * @return a input transformation
   */
  @SuppressWarnings("unchecked")
  public static <T extends Instance> InputTransformation<T, Instance> minMaxNormalization() {
    return (InputTransformation<T, Instance>) MIN_MAX_NORMALIZER;
  }

  @SuppressWarnings("unchecked")
  public static <T extends Instance> InputTransformation<T, Instance> zNormalization() {
    return (InputTransformation<T, Instance>) Z_NORMALIZER;
  }

  public static Transformation<DataFrame, Input<Instance>> toDataset() {
    return (v) -> Inputs::asInput;
  }

  public static <T extends Instance> Transformation<Input<T>, DataFrame> toDataFrame() {
    return new ToDataFrame<>();
  }

  public static <T extends Instance> Transformation<Input<T>, Input<T>> removeIncompleteCases() {
    return new RemoveIncompleteCases<>();
  }

  public static <T> Transformation<Input<T>, DoubleArray> pairwiseDistance(
      Distance<? super T> distance) {
    return new PairwiseDistance<T>(distance);
  }

  private static class MeanImputer<T extends Instance> implements InputTransformation<T, Instance> {
    @Override
    public InputTransformer<T, Instance> fit(Input<T> x) {
      Check.argument(Dataset.isDataset(x));

      int columns = x.getProperty(Dataset.FEATURE_SIZE);
      List<Class<?>> featureTypes = x.getProperty(Dataset.FEATURE_TYPES);
      DoubleArray means = DoubleArray.zeros(columns);
      for (int i = 0; i < x.size(); i++) {
        Instance instance = x.get(i);
        if (instance.size() != columns) {
          throw new IllegalStateException("Illegal instance size.");
        }
        for (int j = 0; j < instance.size(); j++) {
          if (Number.class.isAssignableFrom(featureTypes.get(j))) {
            double value = instance.getDouble(j);
            if (!Is.NA(value)) {
              means.set(j, means.get(j) + (value / x.size()));
            }
          }
        }
      }

      return new MeanImputeInputTransformer<>(means);
    }

    @Override
    public String toString() {
      return "MeanImputer{}";
    }

    private static class MeanImputeInputTransformer<T extends Instance>
        implements InputTransformer<T, Instance> {

      private final DoubleArray means;

      MeanImputeInputTransformer(DoubleArray means) {
        this.means = means;
      }

      @Override
      public String toString() {
        return "MeanImputTransformer{" + "means=" + means + '}';
      }

      @Override
      public Input<Instance> transform(Input<T> x) {
        Check.argument(Dataset.isDataset(x));
        int columns = x.getProperty(Dataset.FEATURE_SIZE);
        Check.state(columns == means.size(), "illegal feature size");

        ArrayInput<Instance> o = x.stream().map(this::transformElement)
            .collect(Collectors.toCollection(() -> new ArrayInput<>(x.getProperties())));
        return Inputs.unmodifiableInput(o);
      }

      @Override
      public Instance transformElement(T x) {
        Check.state(x.size() == means.size(), "illegal feature size");
        ArrayInstance instance = new ArrayInstance();
        for (int i = 0; i < x.size(); i++) {
          if (Number.class.isAssignableFrom(x.getType(i))) {
            double value = x.getDouble(i);
            if (Is.NA(value)) {
              instance.add(means.get(i));
            } else {
              instance.add(value);
            }
          }
        }
        return instance;
      }
    }
  }

  private static class MinMaxNormalizer<T extends Instance>
      implements InputTransformation<T, Instance> {

    @Override
    public InputTransformer<T, Instance> fit(Input<T> x) {
      Check.argument(Dataset.isDataset(x));
      Check.argument(Dataset.isAllNumeric(x), "All numeric dataset required");

      int size = x.getProperty(Dataset.FEATURE_SIZE);
      DoubleArray min = DoubleArray.zeros(size);
      DoubleArray max = DoubleArray.zeros(size);

      for (Instance i : x) {
        for (int j = 0; j < i.size(); j++) {
          double currentMin = min.get(j);
          double currentMax = max.get(j);
          double current = i.getDouble(j);
          if (current > currentMax) {
            max.set(j, current);
          }
          if (current < currentMin) {
            min.set(j, current);
          }
        }
      }


      return new MinMaxNormalizeInputTransformer<>(max, min);
    }

    @Override
    public String toString() {
      return "MinMaxNormalizer";
    }

    private static class MinMaxNormalizeInputTransformer<T extends Instance>
        implements InputTransformer<T, Instance> {

      private final DoubleArray max;
      private final DoubleArray min;

      public MinMaxNormalizeInputTransformer(DoubleArray max, DoubleArray min) {
        Check.argument(max.size() == min.size());
        this.max = max;
        this.min = min;
      }

      @Override
      public Input<Instance> transform(Input<T> x) {
        Check.argument(Dataset.isDataset(x));
        Check.argument(Dataset.isAllNumeric(x), "require numerical features");
        int size = x.getProperty(Dataset.FEATURE_SIZE);
        Check.argument(size == max.size());
        return InputTransformer.super.transform(x);
      }

      @Override
      public Instance transformElement(T x) {
        Check.argument(x.size() == max.size());
        double[] newInstance = new double[x.size()];
        for (int i = 0; i < x.size(); i++) {
          double current = x.getDouble(i);
          if (Is.NA(current)) {
            newInstance[i] = Na.DOUBLE;
          } else {
            double currentMin = min.get(i);
            double currentMax = max.get(i);
            newInstance[i] = (current - currentMin) / (currentMax - currentMin);
          }
        }

        return Instance.wrap(newInstance);
      }

      @Override
      public String toString() {
        return "MinMaxNormalizeInputTransformer{" + "max=" + max + ", min=" + min + '}';
      }
    }
  }

  /**
   * Z normalization is also known as "Normalization to Zero Mean and Unit of Energy" first
   * mentioned in Goldin & Kanellakis. It ensures that all elements of the input vector are
   * transformed into the output vector whose mean is approximately 0 while the standard deviation
   * are in a range close to 1.
   *
   * @author Isak Karlsson
   */
  static class Normalizer<T extends Instance> implements InputTransformation<T, Instance> {

    @Override
    public InputTransformer<T, Instance> fit(Input<T> x) {
      Check.argument(Dataset.isDataset(x));

      List<FastStatistics> statistics = new ArrayList<>();

      int columns = x.getProperty(Dataset.FEATURE_SIZE);
      List<Class<?>> featureType = x.getProperty(Dataset.FEATURE_TYPES);
      for (int i = 0; i < columns; i++) {
        statistics.add(new FastStatistics());
      }

      for (T instance : x) {
        Check.state(instance.size() == columns);
        for (int i = 0; i < instance.size(); i++) {
          if (Is.numeric(featureType.get(i))) {
            double value = instance.getDouble(i);
            if (!Is.NA(value)) {
              statistics.get(i).addValue(value);
            }
          }
        }
      }

      List<StatisticalSummary> summaries =
          statistics.stream().map(FastStatistics::getSummary).collect(Collectors.toList());
      return new NormalizerTransformer<>(featureType, summaries);
    }

    @Override
    public String toString() {
      return "Normalizer";
    }

    private static class NormalizerTransformer<T extends Instance>
        implements InputTransformer<T, Instance> {

      private final List<StatisticalSummary> summaries;
      private final List<Class<?>> features;

      NormalizerTransformer(List<Class<?>> features, List<StatisticalSummary> summaries) {
        this.features = features;
        this.summaries = summaries;
      }

      @Override
      public Instance transformElement(T x) {
        Check.state(x.size() == summaries.size());
        ArrayInstance o = new ArrayInstance();
        for (int i = 0; i < x.size(); i++) {
          double value = x.getDouble(i);
          if (!Is.NA(value) && Is.numeric(features.get(i))) {
            StatisticalSummary summary = summaries.get(i);
            o.add((value - summary.getMean()) / summary.getStandardDeviation());
          } else {
            o.add(x.get(i));
          }
        }

        return o;
      }

      @Override
      public String toString() {
        return "NormalizerTransformer{" + "summaries=" + summaries + '}';
      }
    }
  }

  /**
   * Created by isak on 2016-09-30.
   */
  private static class ToDataFrame<T extends Instance>
      implements Transformation<Input<T>, DataFrame> {

    private final Supplier<? extends DataFrame.Builder> supplier;

    ToDataFrame(Supplier<? extends DataFrame.Builder> supplier) {
      this.supplier = supplier;
    }

    ToDataFrame() {
      this(DataFrame::newBuilder);
    }

    @Override
    public Transformer<Input<T>, DataFrame> fit(Input<T> instances) {
      return (v) -> {
        Check.argument(Dataset.isDataset(instances));
        DataFrame.Builder b = supplier.get();
        List<Class<?>> types = instances.getProperty(Dataset.FEATURE_TYPES);
        List<String> names = instances.getProperty(Dataset.FEATURE_NAMES);
        int features = instances.getProperty(Dataset.FEATURE_SIZE);
        for (int j = 0; j < features; j++) {
          String name = names.get(j);
          Series.Builder builder = Series.Builder.of(types.get(j));
          for (T instance : instances) {
            builder.add(instance.get(j));
          }
          b.setColumn(name, builder);
        }

        return b.build();
      };
    }
  }

  /**
   * Removes incomplete cases, i.e. rows with missing values.
   *
   * @author Isak Karlsson
   */
  private static class RemoveIncompleteCases<T extends Instance>
      implements InputTransformation<T, T> {

    @Override
    public InputTransformer<T, T> fit(Input<T> x) {
      return new InputTransformer<T, T>() {
        @Override
        public Input<T> transform(Input<T> x) {
          Input<T> o = x.stream().filter(instance -> !hasNA(instance))
              .collect(Collectors.toCollection(() -> new ArrayInput<>(x.getProperties())));
          return Inputs.unmodifiableInput(o);
        }

        private boolean hasNA(T instance) {
          for (int i = 0; i < instance.size(); i++) {
            if (Is.NA(instance.get(i))) {
              return true;
            }
          }
          return false;
        }

        @Override
        public T transformElement(T x) {
          if (!hasNA(x))
            return x;
          else {
            return null;
          }
        }
      };
    }
  }

  /**
   * Created by isak on 2016-09-28.
   */
  private static class PairwiseDistance<T> implements Transformation<Input<T>, DoubleArray> {

    private final Distance<? super T> distance;

    PairwiseDistance(Distance<? super T> distance) {
      this.distance = distance;
    }

    @Override
    public Transformer<Input<T>, DoubleArray> fit(Input<T> in) {
      return (v) -> {
        DoubleArray distances = DoubleArray.zeros(in.size(), in.size());
        for (int i = 0; i < in.size(); i++) {
          for (int j = i; j < in.size(); j++) {
            double dist = distance.compute(in.get(i), in.get(j));
            distances.set(i, j, dist);
            distances.set(j, i, dist);
          }
        }
        return distances;
      };
    }
  }
}
