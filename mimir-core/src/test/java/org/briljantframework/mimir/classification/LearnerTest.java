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

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.Collectors;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.parser.CsvParser;
import org.briljantframework.data.series.Series;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.tree.pattern.*;
import org.briljantframework.mimir.classification.tune.Configuration;
import org.briljantframework.mimir.classification.tune.GridSearch;
import org.briljantframework.mimir.classification.tune.Updatables;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.distance.EuclideanDistance;
import org.junit.Test;

/**
 * Created by isak on 11/16/15.
 */
public class LearnerTest {

  public void testPairWiseClassifier() throws FileNotFoundException {
    // DataFrame iris = DataFrames.permute(Datasets.loadIris());
    // DataFrame data = iris.drop("Class").apply(v -> {
    // v.set(v.where(Is::NA), v.mean());
    // return v;
    // });
    // Series labels = iris.get("Class");

    CsvParser parser = new CsvParser(new FileReader("/Users/isak/Tmp/pima-indians-diabetes.txt"));
    parser.getSettings().setSkipRows(1);
    DataFrame dataset = DataFrames.permute(parser.parse(DataFrame::builder));
    String classVariable = "Class variable (0 or 1)";
    DataFrame data = dataset.drop(classVariable).apply(v -> {
      v.set(v.where(Is::NA), v.mean());
      return v;
    });;
    Series labels = dataset.get(classVariable);

    System.out.println(Arrays.sum(0, data.where(Is::NA).intArray()));

    Input<Pair<Series, Series>> trainingInput = new ArrayInput<>();
    Output<Boolean> trainingOutput = new ArrayOutput<>();

    Input<Pair<Series, Series>> testInput = new ArrayInput<>();
    Output<Boolean> testOutput = new ArrayOutput<>();

    int trainSize = Math.round(data.rows() * 0.7f);
    Range trainIdx = Range.of(trainSize);
    Range testIdx = Range.of(trainSize, data.rows());

    createPairs(data.loc().getRow(trainIdx), labels.loc().get(trainIdx), trainingInput,
        trainingOutput);
    createPairs(data.loc().getRow(testIdx), labels.loc().get(testIdx), testInput, testOutput);

    ClassifierValidator<Pair<Series, Series>, Classifier<Pair<Series, Series>, Object>> validator =
        ClassifierValidator.holdoutValidator(testInput, testOutput);


    System.out.println(Outputs.valueCounts(trainingOutput));
    System.out.println(Outputs.valueCounts(testOutput));
    System.out.println(labels.collect(Collectors.valueCounts()));

    RandomPatternForest.Learner<Pair<Series, Series>, Object, Pair<Integer, Double>> learner =
        getPairFeatureLearner(100);
    learner.set(PatternTree.PATTERN_COUNT, 1);
    System.out.println(validator.test(learner, trainingInput, trainingOutput));

  }

  private RandomPatternForest.Learner<Pair<Series, Series>, Boolean, Double> getPairDistanceLearner(
      int size) {
    PatternFactory<Pair<Series, Series>, Double> factory =
        new SamplingPatternFactory<Pair<Series, Series>, Double>() {
          @Override
          protected Double createPattern(Pair<Series, Series> input) {
            return EuclideanDistance.getInstance().compute(input.getLeft().loc(),
                input.getRight().loc());
          }
        };

    PatternDistance<Pair<Series, Series>, Double> distance = (a, b) -> {
      double dist = EuclideanDistance.getInstance().compute(a.getLeft().loc(), a.getRight().loc());
      return Math.abs(dist - b);
    };
    return new RandomPatternForest.Learner<>(factory, distance, size);
  }

  private RandomPatternForest.Learner<Pair<Series, Series>, Object, Pair<Integer, Double>> getPairFeatureLearner(
      int size) {
    PatternFactory<Pair<Series, Series>, Pair<Integer, Double>> factory =
        new SamplingPatternFactory<Pair<Series, Series>, Pair<Integer, Double>>() {
          @Override
          protected Pair<Integer, Double> createPattern(Pair<Series, Series> input) {
            int index = ThreadLocalRandom.current().nextInt(input.getLeft().size());
            double value = computeMeanDiff(input, index);
            return new ImmutablePair<>(index, value / 2);
          }
        };

    PatternDistance<Pair<Series, Series>, Pair<Integer, Double>> distance = (a, b) -> {
      int index = b.getLeft();
      double value = computeMeanDiff(a, index);
      return Math.abs(value - b.getRight());
    };
    return new RandomPatternForest.Learner<>(factory, distance, size);
  }

  private double computeMeanDiff(Pair<Series, Series> input, int index) {
    return input.getLeft().loc().getDouble(index) + input.getRight().loc().getDouble(index);
  }

  private void createPairs(DataFrame data, Series labels, Input<Pair<Series, Series>> input,
      Output<Boolean> output) {
    for (int i = 0; i < data.rows(); i++) {
      for (int j = 0; j < data.rows(); j++) {
        if (i != j) {
          input.add(new ImmutablePair<>(data.loc().getRow(i), data.loc().getRow(j)));
          output.add(Is.equal(labels.loc().get(i), labels.loc().get(j)));
        }
      }
    }
  }

  public void testGridSearch() throws Exception {
    GridSearch<Instance, Object, LogisticRegression<Object>> gridSearch =
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
    LogisticRegression.Learner<Object> learner = new LogisticRegression.Learner<>(1);
    List<Configuration<Object>> tune = gridSearch.tune(learner, x, y);

    tune.sort((a, b) -> -Double.compare(a.getResult().getMeasure("accuracy").mean(),
        b.getResult().getMeasure("accuracy").mean()));

    for (Configuration<Object> f : tune) {
      System.out.println(f.getParameters() + " " + f.getResult().getMeasure("accuracy").mean());
    }

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

  // @Test
  // public void MatThing() throws Exception {
  // DoubleArray data = Arrays
  // .readIdx(new FileInputStream(new File("/home/isak/Tmp/mnist/train-images-idx3-ubyte")));
  // DoubleArray label = Arrays
  // .readIdx(new FileInputStream(new File("/home/isak/Tmp/mnist/train-labels-idx1-ubyte")));
  //
  // Input<DoubleArray> in = new AbstractInput<DoubleArray>() {
  // @Override
  // public DoubleArray get(int index) {
  // return data.select(index);
  // }
  //
  // @Override
  // public int size() {
  // return data.size(0);
  // }
  //
  // @Override
  // public Properties getProperties() {
  // return new Properties();
  // }
  // };
  // Output<?> out = Outputs.copyOf(label);
  //
  // PatternFactory<DoubleArray, DoubleArray> factory =
  // new SamplingPatternFactory<DoubleArray, DoubleArray>() {
  // @Override
  // protected DoubleArray createPattern(DoubleArray input) {
  // int xStart = ThreadLocalRandom.current().nextInt(10);
  // int xEnd = ThreadLocalRandom.current().nextInt(xStart + 5, 29);
  // int yStart = ThreadLocalRandom.current().nextInt(10);
  // int yEnd = ThreadLocalRandom.current().nextInt(yStart + 5, 29);
  // DoubleArray view = input.getView(Range.of(xStart, xEnd), Range.of(yStart, yEnd));
  // return view;
  // }
  // };
  //
  // PatternDistance<DoubleArray, DoubleArray> distance =
  // new PatternDistance<DoubleArray, DoubleArray>() {
  // @Override
  // public double computeDistance(DoubleArray a, DoubleArray b) {
  // double minDistance = Double.POSITIVE_INFINITY;
  // int seriesSize = a.size();
  // int m = b.size();
  // double[] t = new double[m * 2];
  //
  // double ex = 0;
  // double ex2 = 0;
  // for (int i = 0; i < seriesSize; i++) {
  // double d = a.get(i);
  // ex += d;
  // ex2 += d * d;
  // t[i % m] = d;
  // t[(i % m) + m] = d;
  //
  // if (i >= m - 1) {
  // int j = (i + 1) % m;
  // double mean = ex / m;
  // double sigma = FastMath.sqrt(ex2 / m - mean * mean);
  // double dist = distance(b, t, j, m, mean, sigma, minDistance);
  // if (dist < minDistance) {
  // minDistance = dist;
  // }
  //
  // ex -= t[j];
  // ex2 -= t[j] * t[j];
  // }
  // }
  // return Math.sqrt(minDistance / b.size());
  // }
  //
  // double distance(DoubleArray c, double[] t, int j, int m, double mean, double std,
  // double bsf) {
  // double sum = 0;
  // for (int i = 0; i < m && sum < bsf; i++) {
  // double x = normalize(t[i + j], mean, std) - c.get(i);
  // // double x = ((t[i + j] - mean) / std) - c.loc().getAsDouble(i);
  // sum += x * x;
  // }
  // return sum;
  // }
  //
  // public double normalize(double value, double mean, double std) {
  // if (std == 0) {
  // return 0;
  // } else {
  // return (value - mean) / std;
  // }
  // }
  // };
  //
  // RandomPatternForest.Learner<DoubleArray, DoubleArray> rpfl =
  // new RandomPatternForest.Learner<>(factory, distance, 1);
  // rpfl.set(PatternTree.PATTERN_COUNT, 1);
  // rpfl.set(Ensemble.SIZE, 1);
  //
  // ClassifierValidator<DoubleArray, RandomPatternForest<DoubleArray>> validator =
  // ClassifierValidator.splitValidator(0.33);
  // System.out.println(validator.test(rpfl, in, out));
  //
  //
  //
  // }
  //
  // @Test
  // public void testKernelTree() throws Exception {
  // Pair<Input<Instance>, Output<?>> data = loadDataset();
  // Input<Instance> x = data.getLeft();
  // Output<?> y = data.getRight();
  //
  //
  // // PatternFactory<DoubleSequence, DoubleSequence> factory = (inputs, classSet) -> {
  // // int k = ThreadLocalRandom.current().nextInt(2, 10);
  // // KMeansPlusPlusClusterer<DoublePoint> clusterer =
  // // new KMeansPlusPlusClusterer<>(k, 100, new EuclideanDistance(), new JDKRandomGenerator(),
  // // KMeansPlusPlusClusterer.EmptyClusterStrategy.LARGEST_VARIANCE);
  // // List<DoublePoint> points = new ArrayList<>();
  // // for (Example example : classSet) {
  // // points.add(new DoublePoint(toArray(inputs.get(example.getIndex()))));
  // // }
  // // if (points.size() <= k) {
  // // return inputs.get(classSet.getRandomSample().getRandomExample().getIndex());
  // // }
  // // List<CentroidCluster<DoublePoint>> cluster = clusterer.cluster(points);
  // // return DoubleArray.of(
  // // cluster.get(ThreadLocalRandom.current().nextInt(cluster.size())).getCenter().getPoint());
  // // };
  // PatternFactory<DoubleSequence, DoubleSequence> factory =
  // new SamplingPatternFactory<DoubleSequence, DoubleSequence>() {
  // @Override
  // protected DoubleSequence createPattern(DoubleSequence input) {
  // return input;
  // }
  // };
  // PatternDistance<DoubleSequence, DoubleSequence> rbfKernel = this::rbf;
  //
  // Predictor.Learner<DoubleSequence, Object, Classifier<DoubleSequence>> rbfForest =
  // new AbstractLearner<DoubleSequence, Object, Classifier<DoubleSequence>>() {
  //
  // @Override
  // public Classifier<DoubleSequence> fit(Input<? extends DoubleSequence> in, Output<?> out) {
  // int features = in.getProperty(Dataset.FEATURE_SIZE);
  // double[] mean = new double[features];
  // double[] std = new double[features];
  //
  // for (int i = 0; i < features; i++) {
  // FastStatistics statistics = new FastStatistics();
  // for (int j = 0; j < in.size(); j++) {
  // double value = in.get(j).getDouble(i);
  // if (!Is.NA(value)) {
  // statistics.addValue(value);
  // }
  // }
  // mean[i] = statistics.getMean();
  // std[i] = statistics.getStandardDeviation();
  // }
  // RandomPatternForest.Learner<DoubleSequence, DoubleSequence> rfl =
  // new RandomPatternForest.Learner<>(factory, rbfKernel, 100);
  // rfl.set(PatternTree.PATTERN_COUNT, 10);
  //
  // Input<DoubleSequence> xTrans = normalize(in, mean, std);
  // RandomPatternForest<DoubleSequence> model = rfl.fit(xTrans, out);
  // Classifier<DoubleSequence> classifier = new Classifier<DoubleSequence>() {
  //
  // @Override
  // public List<?> getClasses() {
  // return model.getClasses();
  // }
  //
  // @Override
  // public DoubleArray estimate(Input<? extends DoubleSequence> x) {
  // return model.estimate(normalize(x, mean, std));
  // }
  //
  // @Override
  // public DoubleArray estimate(DoubleSequence input) {
  // throw new UnsupportedOperationException();
  // }
  //
  // @Override
  // public Object predict(DoubleSequence doubleSequence) {
  // throw new UnsupportedOperationException();
  // }
  //
  // @Override
  // public Output<Object> predict(Input<? extends DoubleSequence> x) {
  // return model.predict(normalize(x, mean, std));
  // }
  //
  // @Override
  // public Set<Characteristic> getCharacteristics() {
  // return model.getCharacteristics();
  // }
  // };
  //
  // return classifier;
  // };
  // };
  // ClassifierValidator<DoubleSequence, Classifier<DoubleSequence>> v =
  // ClassifierValidator.crossValidator(10);
  // // v.add(new EnsembleEvaluator<>());
  // System.out.println(mean(v.test(rbfForest, x, y).getMeasures()));
  //
  // }
  //
  // private double rbf(DoubleSequence a, DoubleSequence b) {
  // double e = 0;
  // int size = Math.min(a.size(), b.size());
  // for (int i = 0; i < size; i++) {
  // double ad = a.getDouble(i);
  // double bd = b.getDouble(i);
  // if (!Double.isNaN(ad) && !Double.isNaN(bd)) {
  // e += Math.pow(ad - bd, 2);
  // }
  // }
  // return Math.exp(-1 * Math.sqrt(e));
  // }
  //
  // private Input<DoubleSequence> normalize(Input<? extends DoubleSequence> in, double[] mean,
  // double[] std) {
  // Input<DoubleSequence> xTrans = new ArrayInput<>();
  // for (int i = 0; i < in.size(); i++) {
  // DoubleSequence instance = in.get(i);
  //
  // double[] tmp = new double[instance.size()];
  // for (int j = 0; j < instance.size(); j++) {
  // double value = instance.getDouble(j);
  // if (Double.isNaN(value)) {
  // tmp[j] = mean[j];
  // } else if (std[j] == 0) {
  // tmp[j] = 0;
  // } else {
  // tmp[j] = (value - mean[j]) / std[j];
  // }
  // }
  // xTrans.add(DoubleArray.of(tmp));
  // }
  // return xTrans;
  // }
  //
  // private double[] toArray(DoubleSequence instance) {
  // double[] e = new double[instance.size()];
  // for (int i = 0; i < instance.size(); i++) {
  // e[i] = instance.getDouble(i);
  // }
  // return e;
  // }
  //
  // @Test
  // public void testTraditionalRf() throws Exception {
  // Pair<Input<Instance>, Output<?>> data = loadDataset();
  // Input<Instance> x = data.getLeft();
  // Output<?> y = data.getRight();
  //
  // RandomForest.Learner rfl = new RandomForest.Learner(100);
  // ClassifierValidator<Instance, RandomForest> v = ClassifierValidator.crossValidator(10);
  // System.out.println(DataFrames.mean(v.test(rfl, x, y).getMeasures()));
  // }
  //
  // @Test
  // public void testSinglePattern() throws Exception {
  // // DataFrame data = DataFrames.permuteRecords(Datasets.loadIris());
  // // Input<Instance> x = Inputs.newInput(data.drop("Class"));
  // // Output<?> y = Outputs.newOutput(data.get("Class"));
  // Pair<Input<Instance>, Output<?>> data = loadDataset();
  // Input<Instance> x = data.getLeft();
  // Output<?> y = data.getRight();
  //
  //
  // PatternDistance<Instance, Pair<Integer, Object>> zeroOneDistance = (a, b) -> {
  // Object value = b.getValue();
  // int axis = b.getKey();
  // if (Is.numeric(value)) {
  // return Convert.to(Double.class, value).compareTo(a.getDouble(axis)) <= 0 ? 0 : 1;
  // } else {
  // return !Is.equal(value, a.get(axis)) ? 0 : 1;
  // }
  // };
  //
  // SamplingPatternFactory<Instance, Pair<Integer, Object>> featureFactory =
  // new SamplingPatternFactory<Instance, Pair<Integer, Object>>() {
  //
  // @Override
  // protected Pair<Integer, Object> createPattern(Instance input) {
  // int axis = ThreadLocalRandom.current().nextInt(input.size());
  // return new ImmutablePair<>(axis, input.get(axis));
  // }
  // };
  //
  // RandomPatternForest.Learner<Instance, ?> rfl =
  // new RandomPatternForest.Learner<>(featureFactory, zeroOneDistance, 100);
  //
  // ClassifierValidator<Instance, RandomPatternForest<Instance>> v =
  // ClassifierValidator.crossValidator(10);
  //
  // System.out.println(mean(v.test(rfl, x, y).getMeasures()));
  // }
  //
  // @Test
  // public void testTesda2() throws Exception {
  // DataFrame data = DataFrames.permute(Datasets.loadSyntheticControl());
  // Input<Series> x = new ArrayInput<>(data.drop(0).rows());
  // Output<?> y = Outputs.asOutput(data.get(0));
  //
  // ClassifierValidator<Series, RandomPatternForest<Series>> v =
  // ClassifierValidator.splitValidator(0.33);
  //
  // RandomPatternForest.Learner<Series, ?> rfl =
  // new RandomPatternForest.Learner<>(new ShapeletFactory(), new ShapeletDistance(), 100);
  //
  // System.out.println(DataFrames.mean(v.test(rfl, x, y).getMeasures()));
  //
  //
  //
  // // ArrayPrinter.setMinimumTruncateSize(100000);
  // // DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
  // // DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
  // // Series y = iris.get("Class");
  // //
  // // IntArray idx = Arrays.shuffle(Range.of(iris.rows()));
  // // IntArray train = idx.get(Range.of(0, 50));
  // // IntArray cal = idx.get(Range.of(50, 100));
  // // IntArray test = idx.get(Range.of(100, 150));
  // //
  // // ProbabilityEstimateNonconformity.Learner nc =
  // // new ProbabilityEstimateNonconformity.Learner(new RandomForest.Learner(100),
  // // ProbabilityCostFunction.margin());
  // // InductiveConformalClassifier.Learner c = new InductiveConformalClassifier.Learner(nc);
  // // InductiveConformalClassifier icp = c.fit(x.loc().getRecord(train), y.loc().get(train));
  // // icp.calibrate(x.loc().getRecord(cal), y.loc().get(cal));
  // //
  // // DoubleArray prediction = icp.estimate(x.loc().getRecord(test));
  // // System.out.println(Arrays.mean(0, prediction));
  // // ConformalClassifierMeasure m =
  // // new ConformalClassifierMeasure(y.loc().get(test), prediction, 0.9, icp.getClasses());
  // // System.out.println(m.getError());
  //
  // }
  //
  // @Test
  // public void testFit() throws Exception {
  // // Load the iris data set
  // DataFrame iris = DataFrames.permute(Datasets.loadIris());
  //
  // // Remove the class variable from the input data and set each NA value to the column mean
  // DataFrame x = iris.drop("Class").apply(v -> {
  // v.set(v.where(Is::NA), v.mean());
  // return v;
  // });
  //
  // // Get the class variable
  // Series y = iris.get("Class");
  //
  //
  // // Create a classifier learner to use for estimating the non-conformity scores
  // RandomForest.Learner classifier = new RandomForest.Learner(100);
  // ClassifierValidator<Instance, RandomForest> rfv = ClassifierValidator.crossValidator(10);
  //
  // System.out.println(
  // mean(rfv.test(new RandomForest.Learner(100), Inputs.asInput(x), new ArrayOutput<>(y))
  // .getMeasures()));
  //
  // // System.out.println(ClassifierValidator.crossValidator(10).test(classifier, x,
  // // y).getMeasures()
  // // .mean());
  // // Initialize the non-conformity learner using the margin as cost function
  // ProbabilityEstimateNonconformity.Learner<Instance, RandomForest> nc =
  // new ProbabilityEstimateNonconformity.Learner<>(classifier,
  // ProbabilityCostFunction.margin());
  //
  // // Initialize an inductive conformal classifier using the non-conformity learner
  // InductiveConformalClassifier.Learner<Instance> cp = new InductiveConformalClassifier.Learner<>(
  // nc, ClassifierCalibrator.classConditional(), false);
  //
  // // Create a validator for evaluating the validity and efficiency of the conformal classifier.
  // In
  // // this case, we evaluate the classifier using 10-fold cross-validation and 9 significance
  // // levels between 0.1 and 0.1
  // Validator<Instance, Object, InductiveConformalClassifier<Instance>> validator =
  // ConformalClassifierValidator.crossValidator(10, 0.25, DoubleArray.range(0.05, 1.01, 0.05));
  //
  // Result<?> result = validator.test(cp, Inputs.asInput(x), Outputs.asOutput(y));
  //
  // // Get the measures
  // DataFrame measures = result.getMeasures();
  //
  // // Compute the mean of all measures grouped by significance level
  // DataFrame meanPerSignificance = measures
  // .groupBy(Double.class, v -> String.format("%.2f", v), "significance").collect(Series::mean);
  // meanPerSignificance = sort(meanPerSignificance, SortOrder.ASC);
  // System.out.println(meanPerSignificance);
  //
  // // RandomShapeletForest f = forest.fit(x, y);
  //
  // // for (DoubleArray shapelet : f.getImportantShapelets()) {
  // // DoubleArray a = DoubleArray.zeros(x.columns());
  // // for (int i = 0; i < shapelet.size(); i++) {
  // // a.set(shapelet.start() + i, shapelet.loc().getAsDouble(i));
  // // }
  // // System.out.println(shapelet);
  // // }
  // }
  //
  // public static DataFrame loadDatasetExample() throws IOException {
  // // Dataset can be found here: http://www.cs.ucr.edu/~eamonn/time_series_data/
  // String trainFile = "/Users/isak-kar/Downloads/dataset/OliveOil/OliveOil_TRAIN";
  // String testFile = "/Users/isak-kar/Downloads/dataset/OliveOil/OliveOil_TEST";
  // try (DatasetReader train = new MatlabDatasetReader(new FileInputStream(trainFile));
  // DatasetReader test = new MatlabDatasetReader(new FileInputStream(testFile))) {
  // DataFrame.Builder dataset = new MixedDataFrame.Builder();
  // dataset.readAll(train);
  // dataset.readAll(test);
  // return dataset.build();
  // }
  // }
}
