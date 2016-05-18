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
package org.briljantframework.mimir.classification;

import static org.briljantframework.data.dataframe.DataFrames.mean;
import static org.briljantframework.data.dataframe.DataFrames.sort;

import java.io.*;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.FastMath;
import org.briljantframework.DoubleSequence;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataframe.MixedDataFrame;
import org.briljantframework.data.parser.CsvParser;
import org.briljantframework.data.series.Convert;
import org.briljantframework.data.series.Series;
import org.briljantframework.data.series.Vectors;
import org.briljantframework.data.statistics.FastStatistics;
import org.briljantframework.dataset.io.DatasetReader;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.briljantframework.mimir.classification.conformal.ClassifierCalibrator;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ProbabilityCostFunction;
import org.briljantframework.mimir.classification.conformal.ProbabilityEstimateNonconformity;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.mimir.classification.tree.pattern.*;
import org.briljantframework.mimir.classification.tune.Configuration;
import org.briljantframework.mimir.classification.tune.GridSearch;
import org.briljantframework.mimir.classification.tune.Updatables;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.shapelet.ShapeletDistance;
import org.briljantframework.mimir.shapelet.ShapeletFactory;
import org.briljantframework.mimir.supervised.AbstractLearner;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;
import org.junit.Test;

/**
 * Created by isak on 11/16/15.
 */
public class LearnerTest {

  @Test
  public void testGridSearch() throws Exception {
    GridSearch<Instance, Object, LogisticRegression> gridSearch =
        new GridSearch<>(ClassifierValidator.crossValidator(10));
    gridSearch.add(Updatables.enumeration(LogisticRegression.MAX_ITERATIONS, 100));
    gridSearch.add(Updatables.linspace(LogisticRegression.REGULARIZATION, 0.001, 10, 25));

    DataFrame iris = DataFrames.permute(Datasets.loadIris());
    DataFrame replaceNa = iris.drop("Class").apply(v -> {
      v.set(v.where(Is::NA), v.mean());
      return v;
    });
    Series classVector = iris.get("Class");

    Input<Instance> x = Inputs.asInput(replaceNa);
    Output<?> y = Outputs.asOutput(classVector);
    LogisticRegression.Learner learner = new LogisticRegression.Learner(1);
    List<Configuration<Object>> tune = gridSearch.tune(learner, x, y);

    tune.sort((a, b) -> -Double.compare(a.getResult().getMeasure("accuracy").mean(),
        b.getResult().getMeasure("accuracy").mean()));

    for (Configuration<Object> f : tune) {
      System.out.println(f.getParameters() + " " + f.getResult().getMeasure("accuracy").mean());
    }

  }

  @Test
  public void sampleRegion() throws Exception {
    DoubleArray x = Arrays.range(10 * 10).asDoubleArray().reshape(10, 10);
    System.out.println(x);
    System.out.println(x.getView(Range.of(1, 3), Range.of(3, 8)));


  }

  private Pair<Input<Instance>, Output<?>> loadDataset() {
    // DataFrame data = DataFrames.permuteRecords(Datasets.loadIris());
    // Input<Instance> x = Inputs.newInput(data.drop("Class"));
    // Output<?> y = Outputs.newOutput(data.get("Class"));

    try {
      CsvParser parser = new CsvParser(new FileReader("/home/isak/Tmp/madelon.txt"));
      DataFrame df = DataFrames.permute(parser.parse(DataFrame::builder), new Random(123));
      Object column = "class";
      DataFrame x = df.drop(column);
      Series y = df.get(column);

      return new ImmutablePair<>(Inputs.asInput(x), Outputs.asOutput(y));
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void Generate() throws Exception {
    // GridSearch<Instance, Object, RandomForest, RandomForest.Configurator> gridSearch =
    // new GridSearch<>(ClassifierValidator.crossValidator(10));
    // Pair<Input<Instance>, Output<?>> data = loadDataset();
    // Input<Instance> x = data.getLeft();
    // Output<?> y = data.getRight();
    //
    // System.out.println(gridSearch.tune(new RandomForest.Configurator(100), x, y));

  }

  @Test
  public void MatThing() throws Exception {
    DoubleArray data = Arrays
        .readIdx(new FileInputStream(new File("/home/isak/Tmp/mnist/train-images-idx3-ubyte")));
    DoubleArray label = Arrays
        .readIdx(new FileInputStream(new File("/home/isak/Tmp/mnist/train-labels-idx1-ubyte")));

    Input<DoubleArray> in = new AbstractInput<DoubleArray>() {
      @Override
      public DoubleArray get(int index) {
        return data.select(index);
      }

      @Override
      public int size() {
        return data.size(0);
      }

      @Override
      public TypeMap getProperties() {
        return new TypeMap();
      }
    };
    Output<?> out = Outputs.copyOf(label);

    PatternFactory<DoubleArray, DoubleArray> factory =
        new SamplingPatternFactory<DoubleArray, DoubleArray>() {
          @Override
          protected DoubleArray createPattern(DoubleArray input) {
            int xStart = ThreadLocalRandom.current().nextInt(10);
            int xEnd = ThreadLocalRandom.current().nextInt(xStart + 5, 29);
            int yStart = ThreadLocalRandom.current().nextInt(10);
            int yEnd = ThreadLocalRandom.current().nextInt(yStart + 5, 29);
            DoubleArray view = input.getView(Range.of(xStart, xEnd), Range.of(yStart, yEnd));
            return view;
          }
        };

    PatternDistance<DoubleArray, DoubleArray> distance =
        new PatternDistance<DoubleArray, DoubleArray>() {
          @Override
          public double computeDistance(DoubleArray a, DoubleArray b) {
            double minDistance = Double.POSITIVE_INFINITY;
            int seriesSize = a.size();
            int m = b.size();
            double[] t = new double[m * 2];

            double ex = 0;
            double ex2 = 0;
            for (int i = 0; i < seriesSize; i++) {
              double d = a.get(i);
              ex += d;
              ex2 += d * d;
              t[i % m] = d;
              t[(i % m) + m] = d;

              if (i >= m - 1) {
                int j = (i + 1) % m;
                double mean = ex / m;
                double sigma = FastMath.sqrt(ex2 / m - mean * mean);
                double dist = distance(b, t, j, m, mean, sigma, minDistance);
                if (dist < minDistance) {
                  minDistance = dist;
                }

                ex -= t[j];
                ex2 -= t[j] * t[j];
              }
            }
            return Math.sqrt(minDistance / b.size());
          }

          double distance(DoubleArray c, double[] t, int j, int m, double mean, double std,
              double bsf) {
            double sum = 0;
            for (int i = 0; i < m && sum < bsf; i++) {
              double x = normalize(t[i + j], mean, std) - c.get(i);
              // double x = ((t[i + j] - mean) / std) - c.loc().getAsDouble(i);
              sum += x * x;
            }
            return sum;
          }

          public double normalize(double value, double mean, double std) {
            if (std == 0) {
              return 0;
            } else {
              return (value - mean) / std;
            }
          }
        };

    RandomPatternForest.Learner<DoubleArray, DoubleArray> rpfl =
        new RandomPatternForest.Learner<>(factory, distance, 1);
    rpfl.set(PatternTree.PATTERN_COUNT, 1);
    rpfl.set(Ensemble.SIZE, 1);

    ClassifierValidator<DoubleArray, RandomPatternForest<DoubleArray>> validator =
        ClassifierValidator.splitValidator(0.33);
    System.out.println(validator.test(rpfl, in, out));



  }

  @Test
  public void testKernelTree() throws Exception {
    Pair<Input<Instance>, Output<?>> data = loadDataset();
    Input<Instance> x = data.getLeft();
    Output<?> y = data.getRight();


    // PatternFactory<DoubleSequence, DoubleSequence> factory = (inputs, classSet) -> {
    // int k = ThreadLocalRandom.current().nextInt(2, 10);
    // KMeansPlusPlusClusterer<DoublePoint> clusterer =
    // new KMeansPlusPlusClusterer<>(k, 100, new EuclideanDistance(), new JDKRandomGenerator(),
    // KMeansPlusPlusClusterer.EmptyClusterStrategy.LARGEST_VARIANCE);
    // List<DoublePoint> points = new ArrayList<>();
    // for (Example example : classSet) {
    // points.add(new DoublePoint(toArray(inputs.get(example.getIndex()))));
    // }
    // if (points.size() <= k) {
    // return inputs.get(classSet.getRandomSample().getRandomExample().getIndex());
    // }
    // List<CentroidCluster<DoublePoint>> cluster = clusterer.cluster(points);
    // return DoubleArray.of(
    // cluster.get(ThreadLocalRandom.current().nextInt(cluster.size())).getCenter().getPoint());
    // };
    PatternFactory<DoubleSequence, DoubleSequence> factory =
        new SamplingPatternFactory<DoubleSequence, DoubleSequence>() {
          @Override
          protected DoubleSequence createPattern(DoubleSequence input) {
            return input;
          }
        };
    PatternDistance<DoubleSequence, DoubleSequence> rbfKernel = this::rbf;

    Predictor.Learner<DoubleSequence, Object, Classifier<DoubleSequence>> rbfForest =
        new AbstractLearner<DoubleSequence, Object, Classifier<DoubleSequence>>() {

          @Override
          public Classifier<DoubleSequence> fit(Input<? extends DoubleSequence> in, Output<?> out) {
            int features = in.getProperty(Dataset.FEATURE_SIZE);
            double[] mean = new double[features];
            double[] std = new double[features];

            for (int i = 0; i < features; i++) {
              FastStatistics statistics = new FastStatistics();
              for (int j = 0; j < in.size(); j++) {
                double value = in.get(j).getDouble(i);
                if (!Is.NA(value)) {
                  statistics.addValue(value);
                }
              }
              mean[i] = statistics.getMean();
              std[i] = statistics.getStandardDeviation();
            }
            RandomPatternForest.Learner<DoubleSequence, DoubleSequence> rfl =
                new RandomPatternForest.Learner<>(factory, rbfKernel, 100);
            rfl.set(PatternTree.PATTERN_COUNT, 10);

            Input<DoubleSequence> xTrans = normalize(in, mean, std);
            RandomPatternForest<DoubleSequence> model = rfl.fit(xTrans, out);
            Classifier<DoubleSequence> classifier = new Classifier<DoubleSequence>() {

              @Override
              public List<?> getClasses() {
                return model.getClasses();
              }

              @Override
              public DoubleArray estimate(Input<? extends DoubleSequence> x) {
                return model.estimate(normalize(x, mean, std));
              }

              @Override
              public DoubleArray estimate(DoubleSequence input) {
                throw new UnsupportedOperationException();
              }

              @Override
              public Object predict(DoubleSequence doubleSequence) {
                throw new UnsupportedOperationException();
              }

              @Override
              public Output<Object> predict(Input<? extends DoubleSequence> x) {
                return model.predict(normalize(x, mean, std));
              }

              @Override
              public Set<Characteristic> getCharacteristics() {
                return model.getCharacteristics();
              }
            };

            return classifier;
          };
        };
    ClassifierValidator<DoubleSequence, Classifier<DoubleSequence>> v =
        ClassifierValidator.crossValidator(10);
    // v.add(new EnsembleEvaluator<>());
    System.out.println(mean(v.test(rbfForest, x, y).getMeasures()));

  }

  private double rbf(DoubleSequence a, DoubleSequence b) {
    double e = 0;
    int size = Math.min(a.size(), b.size());
    for (int i = 0; i < size; i++) {
      double ad = a.getDouble(i);
      double bd = b.getDouble(i);
      if (!Double.isNaN(ad) && !Double.isNaN(bd)) {
        e += Math.pow(ad - bd, 2);
      }
    }
    return Math.exp(-1 * Math.sqrt(e));
  }

  private Input<DoubleSequence> normalize(Input<? extends DoubleSequence> in, double[] mean,
      double[] std) {
    Input<DoubleSequence> xTrans = new ArrayInput<>();
    for (int i = 0; i < in.size(); i++) {
      DoubleSequence instance = in.get(i);

      double[] tmp = new double[instance.size()];
      for (int j = 0; j < instance.size(); j++) {
        double value = instance.getDouble(j);
        if (Double.isNaN(value)) {
          tmp[j] = mean[j];
        } else if (std[j] == 0) {
          tmp[j] = 0;
        } else {
          tmp[j] = (value - mean[j]) / std[j];
        }
      }
      xTrans.add(DoubleArray.of(tmp));
    }
    return xTrans;
  }

  private double[] toArray(DoubleSequence instance) {
    double[] e = new double[instance.size()];
    for (int i = 0; i < instance.size(); i++) {
      e[i] = instance.getDouble(i);
    }
    return e;
  }

  @Test
  public void testTraditionalRf() throws Exception {
    Pair<Input<Instance>, Output<?>> data = loadDataset();
    Input<Instance> x = data.getLeft();
    Output<?> y = data.getRight();

    RandomForest.Learner rfl = new RandomForest.Learner(100);
    ClassifierValidator<Instance, RandomForest> v = ClassifierValidator.crossValidator(10);
    System.out.println(DataFrames.mean(v.test(rfl, x, y).getMeasures()));
  }

  @Test
  public void testSinglePattern() throws Exception {
    // DataFrame data = DataFrames.permuteRecords(Datasets.loadIris());
    // Input<Instance> x = Inputs.newInput(data.drop("Class"));
    // Output<?> y = Outputs.newOutput(data.get("Class"));
    Pair<Input<Instance>, Output<?>> data = loadDataset();
    Input<Instance> x = data.getLeft();
    Output<?> y = data.getRight();


    PatternDistance<Instance, Pair<Integer, Object>> zeroOneDistance = (a, b) -> {
      Object value = b.getValue();
      int axis = b.getKey();
      if (Is.numeric(value)) {
        return Convert.to(Double.class, value).compareTo(a.getDouble(axis)) <= 0 ? 0 : 1;
      } else {
        return !Is.equal(value, a.get(axis)) ? 0 : 1;
      }
    };

    SamplingPatternFactory<Instance, Pair<Integer, Object>> featureFactory =
        new SamplingPatternFactory<Instance, Pair<Integer, Object>>() {

          @Override
          protected Pair<Integer, Object> createPattern(Instance input) {
            int axis = ThreadLocalRandom.current().nextInt(input.size());
            return new ImmutablePair<>(axis, input.get(axis));
          }
        };

    RandomPatternForest.Learner<Instance, ?> rfl =
        new RandomPatternForest.Learner<>(featureFactory, zeroOneDistance, 100);

    ClassifierValidator<Instance, RandomPatternForest<Instance>> v =
        ClassifierValidator.crossValidator(10);

    System.out.println(mean(v.test(rfl, x, y).getMeasures()));
  }

  @Test
  public void testTesda2() throws Exception {
    DataFrame data = DataFrames.permute(Datasets.loadSyntheticControl());
    Input<Series> x = new ArrayInput<>(data.drop(0).rows());
    Output<?> y = Outputs.asOutput(data.get(0));

    ClassifierValidator<Series, RandomPatternForest<Series>> v =
        ClassifierValidator.splitValidator(0.33);

    RandomPatternForest.Learner<Series, ?> rfl =
        new RandomPatternForest.Learner<>(new ShapeletFactory(), new ShapeletDistance(), 100);

    System.out.println(DataFrames.mean(v.test(rfl, x, y).getMeasures()));



    // ArrayPrinter.setMinimumTruncateSize(100000);
    // DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    // DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    // Series y = iris.get("Class");
    //
    // IntArray idx = Arrays.shuffle(Range.of(iris.rows()));
    // IntArray train = idx.get(Range.of(0, 50));
    // IntArray cal = idx.get(Range.of(50, 100));
    // IntArray test = idx.get(Range.of(100, 150));
    //
    // ProbabilityEstimateNonconformity.Learner nc =
    // new ProbabilityEstimateNonconformity.Learner(new RandomForest.Learner(100),
    // ProbabilityCostFunction.margin());
    // InductiveConformalClassifier.Learner c = new InductiveConformalClassifier.Learner(nc);
    // InductiveConformalClassifier icp = c.fit(x.loc().getRecord(train), y.loc().get(train));
    // icp.calibrate(x.loc().getRecord(cal), y.loc().get(cal));
    //
    // DoubleArray prediction = icp.estimate(x.loc().getRecord(test));
    // System.out.println(Arrays.mean(0, prediction));
    // ConformalClassifierMeasure m =
    // new ConformalClassifierMeasure(y.loc().get(test), prediction, 0.9, icp.getClasses());
    // System.out.println(m.getError());

  }

  @Test
  public void testFit() throws Exception {
    // Load the iris data set
    DataFrame iris = DataFrames.permute(Datasets.loadIris());

    // Remove the class variable from the input data and set each NA value to the column mean
    DataFrame x = iris.drop("Class").apply(v -> {
      v.set(v.where(Is::NA), v.mean());
      return v;
    });

    // Get the class variable
    Series y = iris.get("Class");


    // Create a classifier learner to use for estimating the non-conformity scores
    RandomForest.Learner classifier = new RandomForest.Learner(100);
    ClassifierValidator<Instance, RandomForest> rfv = ClassifierValidator.crossValidator(10);

    System.out.println(mean(
        rfv.test(new RandomForest.Learner(100), Inputs.asInput(x), new ArrayOutput<>(y))
            .getMeasures()));

    // System.out.println(ClassifierValidator.crossValidator(10).test(classifier, x,
    // y).getMeasures()
    // .mean());
    // Initialize the non-conformity learner using the margin as cost function
    ProbabilityEstimateNonconformity.Learner<Instance, RandomForest> nc =
        new ProbabilityEstimateNonconformity.Learner<>(classifier,
            ProbabilityCostFunction.margin());

    // Initialize an inductive conformal classifier using the non-conformity learner
    InductiveConformalClassifier.Learner<Instance> cp = new InductiveConformalClassifier.Learner<>(
        nc, ClassifierCalibrator.classConditional(), false);

    // Create a validator for evaluating the validity and efficiency of the conformal classifier. In
    // this case, we evaluate the classifier using 10-fold cross-validation and 9 significance
    // levels between 0.1 and 0.1
    Validator<Instance, Object, InductiveConformalClassifier<Instance>> validator =
        ConformalClassifierValidator.crossValidator(10, 0.25, DoubleArray.range(0.05, 1.01, 0.05));

    Result<?> result = validator.test(cp, Inputs.asInput(x), Outputs.asOutput(y));

    // Get the measures
    DataFrame measures = result.getMeasures();

    // Compute the mean of all measures grouped by significance level
    DataFrame meanPerSignificance =
        measures.groupBy(Double.class, v -> String.format("%.2f", v), "significance")
            .collect(Vectors::mean);
    meanPerSignificance = sort(meanPerSignificance, SortOrder.ASC);
    System.out.println(meanPerSignificance);

    // RandomShapeletForest f = forest.fit(x, y);

    // for (DoubleArray shapelet : f.getImportantShapelets()) {
    // DoubleArray a = DoubleArray.zeros(x.columns());
    // for (int i = 0; i < shapelet.size(); i++) {
    // a.set(shapelet.start() + i, shapelet.loc().getAsDouble(i));
    // }
    // System.out.println(shapelet);
    // }
  }

  public static DataFrame loadDatasetExample() throws IOException {
    // Dataset can be found here: http://www.cs.ucr.edu/~eamonn/time_series_data/
    String trainFile = "/Users/isak-kar/Downloads/dataset/OliveOil/OliveOil_TRAIN";
    String testFile = "/Users/isak-kar/Downloads/dataset/OliveOil/OliveOil_TEST";
    try (DatasetReader train = new MatlabDatasetReader(new FileInputStream(trainFile));
        DatasetReader test = new MatlabDatasetReader(new FileInputStream(testFile))) {
      DataFrame.Builder dataset = new MixedDataFrame.Builder();
      dataset.readAll(train);
      dataset.readAll(test);
      return dataset.build();
    }
  }
}
