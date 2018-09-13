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
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.Na;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.DoubleSeries;
import org.briljantframework.data.series.Series;
import org.briljantframework.data.statistics.FastStatistics;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;
import org.briljantframework.mimir.distance.Distance;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.InstanceBuilder;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;

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

  /*
   * public static Transformation<DataFrame, Input<Instance>> toDataset() { return (v) ->
   * Inputs::asInput; }
   */

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
      Check.argument(x.getSchema() instanceof MultidimensionalSchema);
      MultidimensionalSchema schema = (MultidimensionalSchema) x.getSchema();
      DoubleArray means = DoubleArray.zeros(schema.numericalAttributes());
      for (int i = 0; i < x.size(); i++) {
        Instance instance = x.get(i);
        for (int j = 0; j < instance.numericalAttributes(); j++) {
          double value = instance.getNumericalAttribute(j);
          if (!Is.NA(value)) {
            means.set(j, means.get(j) + (value / x.size()));
          }
        }
      }

      return new MeanImputeInputTransformer<>(schema, means);
    }

    @Override
    public String toString() {
      return "MeanImputer{}";
    }

    private static class MeanImputeInputTransformer<T extends Instance>
        implements InputTransformer<T, Instance> {

      private final DoubleArray means;
      private final MultidimensionalSchema schema;

      MeanImputeInputTransformer(MultidimensionalSchema schema, DoubleArray means) {
        this.schema = schema;
        this.means = means;
      }

      @Override
      public String toString() {
        return "MeanImputTransformer{" + "means=" + means + '}';
      }

      @Override
      public Input<Instance> transform(Input<T> x) {
        Check.argument(schema.equals(x.getSchema()), "illegal schema");
        MultidimensionalSchema schema = (MultidimensionalSchema) x.getSchema();
        Check.state(schema.numericalAttributes() == means.size(), "illegal feature size");

        Input<Instance> o = x.stream().map(this::transformElement)
            .collect(Collectors.toCollection(schema::newInput));
        return Inputs.unmodifiableInput(o);
      }

      @Override
      public Instance transformElement(T x) {
        Check.argument(schema.isValid(x), "illegal instance");
        InstanceBuilder ib = schema.newInstance();
        for (int i = 0; i < x.numericalAttributes(); i++) {
          double numericalAttribute = x.getNumericalAttribute(i);
          if (Is.NA(numericalAttribute)) {
            ib.set(i, means.get(i));
          } else {
            ib.set(i, numericalAttribute);
          }
        }
        for (int i = 0; i < x.categoricalAttributes(); i++) {
          ib.set(i, x.getCategoricalAttribute(i));
        }
        return ib.build();
      }
    }
  }

  private static class MinMaxNormalizer<T extends Instance>
      implements InputTransformation<T, Instance> {

    @Override
    public InputTransformer<T, Instance> fit(Input<T> x) {
      // Check.argument(Dataset.isDataset(x));
      // Check.argument(Dataset.isAllNumeric(x), "All numeric dataset required");

      // int size = x.getProperty(Dataset.FEATURE_SIZE);
      DoubleArray min = DoubleArray.zeros(0);
      DoubleArray max = DoubleArray.zeros(0); // todo
      //
      // for (Instance i : x) {
      // for (int j = 0; j < i.size(); j++) {
      // double currentMin = min.get(j);
      // double currentMax = max.get(j);
      // double current = i.getDouble(j);
      // if (current > currentMax) {
      // max.set(j, current);
      // }
      // if (current < currentMin) {
      // min.set(j, current);
      // }
      // }
      // }


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
        // Check.argument(Dataset.isDataset(x));
        // Check.argument(Dataset.isAllNumeric(x), "require numerical features");
        // int size = x.getProperty(Dataset.FEATURE_SIZE);
        int size = 0; // todo
        Check.argument(size == max.size());
        return null;
      }

      @Override
      public Instance transformElement(T x) {
        // Check.argument(x.size() == max.size());
        // double[] newInstance = new double[x.size()];
        // for (int i = 0; i < x.size(); i++) {
        // double current = x.getDouble(i);
        // if (Is.NA(current)) {
        // newInstance[i] = Na.DOUBLE;
        // } else {
        // double currentMin = min.get(i);
        // double currentMax = max.get(i);
        // newInstance[i] = (current - currentMin) / (currentMax - currentMin);
        // }
        // }

        return null; // todo
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
      Check.argument(x.getSchema() instanceof MultidimensionalSchema);
      MultidimensionalSchema schema = (MultidimensionalSchema) x.getSchema();
      List<FastStatistics> statistics = new ArrayList<>(schema.numericalAttributes());

      for (int i = 0; i < schema.numericalAttributes(); i++) {
        statistics.add(new FastStatistics());
      }

      for (T instance : x) {
        for (int i = 0; i < instance.numericalAttributes(); i++) {
          double value = instance.getNumericalAttribute(i);
          if (!Is.NA(value)) {
            statistics.get(i).addValue(value);
          }
        }
      }

      List<StatisticalSummary> summaries =
          statistics.stream().map(FastStatistics::getSummary).collect(Collectors.toList());
      return new NormalizerTransformer<>(schema, summaries);
    }

    @Override
    public String toString() {
      return "Normalizer";
    }

    private static class NormalizerTransformer<T extends Instance>
        implements InputTransformer<T, Instance> {

      private final List<StatisticalSummary> summaries;
      private final MultidimensionalSchema schema;

      NormalizerTransformer(MultidimensionalSchema schema, List<StatisticalSummary> summaries) {
        this.schema = schema;
        this.summaries = summaries;
      }

      @Override
      public Input<Instance> transform(Input<T> x) {
        return x.stream().map(this::transformElement)
            .collect(Collectors.toCollection(schema::newInput));
      }

      @Override
      public Instance transformElement(T x) {
        Check.argument(schema.isValid(x));
        InstanceBuilder ib = schema.newInstance();

        for (int i = 0; i < x.numericalAttributes(); i++) {
          double value = x.getNumericalAttribute(i);
          if (!Is.NA(value)) {
            StatisticalSummary summary = summaries.get(i);
            ib.set(i, (value - summary.getMean()) / summary.getStandardDeviation());
          } else {
            ib.set(i, Na.DOUBLE);
          }
        }
        return ib.setAll(x.getCategoricalAttributes()).build();
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
        Check.argument(instances.getSchema() instanceof MultidimensionalSchema);
        DataFrame.Builder b = supplier.get();
        MultidimensionalSchema schema = (MultidimensionalSchema) instances.getSchema();
        for (int i = 0; i < schema.attributes(); i++) {
          Series.Builder builder;
          if (schema.isNumericalAttribute(i)) {
            builder = new DoubleSeries.Builder(instances.size());
            for (T instance : instances) {
              builder.addDouble(instance.getNumericalAttribute(i));
            }
          } else {
            builder = Series.Builder.of(Object.class);
            for (T instance : instances) {
              builder.add(instance.getCategoricalAttribute(i));
            }
          }
          b.setColumn(schema.getAttributeName(i), builder);
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
              .collect(Collectors.toCollection(() -> x.getSchema().newInput()));
          return Inputs.unmodifiableInput(o);
        }

        private boolean hasNA(T instance) {
          for (int i = 0; i < instance.numericalAttributes(); i++) {
            if (Is.NA(instance.getNumericalAttribute(i))) {
              return true;
            }
          }
          for (int i = 0; i < instance.categoricalAttributes(); i++) {
            if (Is.NA(instance.getCategoricalAttribute(i))) {
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
