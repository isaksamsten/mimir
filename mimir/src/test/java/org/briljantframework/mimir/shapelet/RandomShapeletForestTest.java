/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.briljantframework.mimir.shapelet;

import static org.briljantframework.mimir.classification.ClassifierValidator.crossValidator;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.index.Index;
import org.briljantframework.data.reader.EntryReader;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.VectorType;
import org.briljantframework.data.vector.Vectors;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.briljantframework.dataset.io.SequenceDatasetReader;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.classification.RandomShapeletForest;
import org.briljantframework.mimir.classification.conformal.ClassifierNonconformity;
import org.briljantframework.mimir.classification.conformal.ConformalClassifier;
import org.briljantframework.mimir.classification.conformal.DistanceNonconformity;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.evaluation.partition.Partition;
import org.briljantframework.mimir.evaluation.partition.SplitPartitioner;
import org.briljantframework.mimir.supervised.Predictor;
import org.junit.Ignore;
import org.junit.Test;

@Ignore
public class RandomShapeletForestTest {

  // @Test
  // public void testLOOCV() throws Exception {
  // String name = "BirdChicken";
  // String trainFile = String.format("/Users/isak-kar/Downloads/dataset3/%s/%s.arff", name, name);
  // try (DataInputStream train = new ArffInputStream(new FileInputStream(trainFile))) {
  // DataFrame trainingSet = MixedDataFrame.read(train);
  // Transformer znorm = new DataSeriesNormalization();
  // // DataFrame xTrain =
  // // znorm.transform(new DataSeriesCollection.Builder(DoubleVector.TYPE)
  // // .stack(0, trainingSet.drop(trainingSet.columns() - 1))
  // // .build());
  // Vector yTrain = Convert.toStringVector(trainingSet.get(trainingSet.columns() - 1));
  //
  // RandomShapeletForest f =
  // RandomShapeletForest
  // .withSize(100)
  // .withInspectedShapelets(50)
  // .withLowerLength(0.025)
  // .withUpperLength(0.3)
  // .withAssessment(ShapeletTree.Assessment.FSTAT)
  // .build();
  // List<Evaluator> evaluatorList = Evaluator.getDefaultClassificationEvaluators();
  // evaluatorList.add(new Evaluator() {
  // private int fold = 0;
  //
  // @Override
  // public void accept(EvaluationContextImpl ctx) {
  // System.out.printf("Fold %d\n", fold++);
  // }
  // });
  // Result re = Validators.leaveOneOutValidation().test(f, xTrain, yTrain);
  // System.out.println(re);
  // }
  // throw new UnsupportedOperationException();

  // }

  // @Test
  // public void testClassifiy2() throws Exception {
  // String name = "DP_Middle";
  // String trainFile =
  // String.format("/Users/isak-kar/Downloads/dataset3/%s/%s_TRAIN.arff", name, name);
  // String testFile =
  // String.format("/Users/isak-kar/Downloads/dataset3/%s/%s_TEST.arff", name, name);
  // try (DataInputStream train = new ArffInputStream(new FileInputStream(trainFile));
  // DataInputStream test = new ArffInputStream(new FileInputStream(testFile))) {
  // train.readColumnIndex(); // TODO!
  // DataFrame trainingSet =
  // new MixedDataFrame.Builder(train.readColumnTypes()).read(train)
  // .build();
  //
  // test.readColumnIndex(); // TODO:
  // DataFrame validationSet =
  // new MixedDataFrame.Builder(test.readColumnTypes()).read(test)
  // .build();
  //
  // Transformer znorm = new DataSeriesNormalization();
  // DataFrame xTrain =
  // znorm.transform(new DataSeriesCollection.Builder(DoubleVector.TYPE).stack(
  // 0, trainingSet.drop(trainingSet.columns() - 1)).build());
  // Vector yTrain = Convert.toStringVector(trainingSet.get(trainingSet.columns() - 1));
  //
  // DataFrame xTest =
  // znorm.transform(new DataSeriesCollection.Builder(DoubleVector.TYPE).stack(
  // 0, validationSet.drop(validationSet.columns() - 1)).build());
  // Vector yTest = Convert.toStringVector(validationSet.get(validationSet.columns() - 1));
  //
  // long start = System.nanoTime();
  // DoubleArray upper = Arrays.array(new double[]{0.05, 0.1, 0.3, 0.5, 0.7, 1});
  // IntArray sizes = Arrays.array(new int[]{100});
  // // IntMatrix sizes = IntMatrix.of(500);
  // System.out.println("Size,Correlation,Strength,Quality,Expected Error,"
  // + "Accuracy,OOB Accuracy,Variance,Bias,Brier,Depth");
  // for (int i = 0; i < sizes.size(); i++) {
  // RandomShapeletForest forest =
  // RandomShapeletForest.withSize(1000).withInspectedShapelets(sizes.get(i))
  // .withLowerLength(0.025).withUpperLength(0.5)
  // // .withSampleMode(ShapeletTree.SampleMode.RANDOMIZE)
  // .withAssessment(ShapeletTree.Assessment.FSTAT).build();
  // Result result = HoldoutValidator.withHoldout(xTest, yTest).test(forest, xTrain, yTrain);
  // // System.out.println(result);
  // System.out.println(sizes.get(i) + ", "
  // + result.getAverage(Ensemble.Correlation.class) + ", "
  // + result.getAverage(Ensemble.Strength.class) + ", "
  // + result.getAverage(Ensemble.Quality.class) + ", "
  // + result.getAverage(Ensemble.ErrorBound.class) + ", "
  // + result.getAverage(Accuracy.class) + ", "
  // + result.getAverage(Ensemble.OobAccuracy.class) + ", "
  // + result.getAverage(Ensemble.Variance.class) + ", "
  // + result.getAverage(Ensemble.MeanSquareError.class) + ", "
  // + result.getAverage(Brier.class) + ", "
  // + result.getAverage(RandomShapeletForest.Depth.class));
  // }
  // System.out.println((System.nanoTime() - start) / 1e6);
  //
  // }
  // }


  @Test
  public void testSynticControl() throws Exception {
    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new dDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/SwedishLeaf/SwedishLeaf_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/SwedishLeaf/SwedishLeaf_TEST")));

    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/Gun_Point/Gun_Point_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/Gun_Point/Gun_Point_TEST")));
    //
    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset2/SonyAIBORobotSurfaceII/SonyAIBORobotSurfaceII_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset2/SonyAIBORobotSurfaceII/SonyAIBORobotSurfaceII_TEST")));
    //
    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset2/ECGFiveDays/ECGFiveDays_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset2/ECGFiveDays/ECGFiveDays_TEST")));

    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/Coffee/Coffee_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/Coffee/Coffee_TEST")));

    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream("/Users/isak-kar/Downloads/dataset2/wafer/wafer_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream("/Users/isak-kar/Downloads/dataset2/wafer/wafer_TEST")));

    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset2/MoteStrain/MoteStrain_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset2/MoteStrain/MoteStrain_TEST")));
    //
    // DataFrame train = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/Two_Patterns/Two_Patterns_TRAIN")));
    // DataFrame test = Datasets.load(
    // (i) -> new DataSeriesCollection.Builder(double.class), new MatlabDatasetReader(
    // new FileInputStream(
    // "/Users/isak-kar/Downloads/dataset/Two_Patterns/Two_Patterns_TEST")));

    String fileName = "synthetic_control";
    String path = "/Users/isak-kar/Downloads/dataset";
    DataFrame train =
        DataFrames
            .permuteRecords(
                Datasets.load((i) -> new DataSeriesCollection.Builder(double.class),
                    new MatlabDatasetReader(new FileInputStream(
                        String.format("%s/%s/%s_TRAIN", path, fileName, fileName)))),
        new Random(123));
    DataFrame test = Datasets.load((i) -> new DataSeriesCollection.Builder(double.class),
        new MatlabDatasetReader(
            new FileInputStream(String.format("%s/%s/%s_TEST", path, fileName, fileName))));

    train.setColumnIndex(Index.range(train.columns()));
    test.setColumnIndex(Index.range(test.columns()));
    System.out.println(test.get(0).valueCounts());
    System.out.println(train.get(0).valueCounts());

    // System.out.println(train.head(5));
    // System.out.println(train.getRecord(0).asList(Double.class).subList(1, 85));
    // System.out.println(train.drop(0).getRecord(0).asList(Double.class));
    // System.out.println(train.drop(0).columns());
    // Classifier rsf =
    // RandomShapeletForest.withSize(100).withAssessment(ShapeletTree.Assessment.IG)
    // .withInspectedShapelets(100).withUpperLength(1).build();
    // Classifier knn = KNearestNeighbors.withNeighbors(1).build();
    // System.out.println(HoldoutValidator.withHoldout(test.drop(0), test.get(0)).test(knn,
    // train.drop(0), train.get(0)).get(ErrorRate.class));

    // .getAverage(Accuracy.class));
    // CBF = 0.3
    // synthetic_control = 0.1
    // Gun_Point = 0.3
    // Mote_strain = 0.5
    Partition trainPart =
        new SplitPartitioner(0.1).partition(train.drop(0), train.get(0)).iterator().next();

    ClassifierNonconformity.Learner nc = new DistanceNonconformity.Learner(1);
    // Nonconformity.Learner nc =
    // new ProbabilityEstimateNonconformity.Learner(
    // new
    // RandomShapeletForest.Configurator(100).setAssessment(ShapeletTree.Learner.Assessment.IG).configure(),
    // new Margin());
    InductiveConformalClassifier.Learner learner = new InductiveConformalClassifier.Learner(nc);
    InductiveConformalClassifier classifier =
        learner.fit(trainPart.getTrainingData(), trainPart.getTrainingTarget());
    classifier.calibrate(trainPart.getValidationData(), trainPart.getValidationTarget());
    testEarlyClassification(test, classifier);
  }

  private void testEarlyClassification(DataFrame test, ConformalClassifier classifier) {
    Vector c = classifier.getClasses();
    DataFrame x = test.drop(0);
    Vector y = test.get(0);

    IntArray d = Arrays.newIntArray(x.rows());
    double correct = 0;
    for (int i = 0; i < x.rows(); i++) {
      if (i % 100 == 0) {
        System.out.printf("Processing test instance %d/%d\n", i, x.rows());
      }
      Vector record = x.loc().getRecord(i);
      Object trueLabel = y.loc().get(Object.class, i);
      boolean found = false;
      for (int j = 5; j < record.size() && !found; j++) {
        double minSignificance = 0.05;
        double minConfidence = 0.95;
//        DoubleArray estimates = classifier.estimate(record.select(0, j));
//        int prediction = Arrays.argmax(estimates);
//        double credibility = estimates.get(prediction);
//        double confidence = 1 - maxnot(estimates, prediction);
//         System.out.println(confidence + " " + credibility + " from " + estimates);
//        if (confidence >= minConfidence && credibility >= minSignificance) {
//          d.set(i, j);
//          correct += Is.equal(c.loc().get(prediction), trueLabel) ? 1 : 0;
//          found = true;
//        }
      }

      // Classify it once all time has passed
//      if (!found) {
//        Object prediction = c.loc().get(Arrays.argmax(classifier.estimate(record)));
//        correct += Is.equal(trueLabel, prediction) ? 1 : 0;
//        d.set(i, record.size());
//      }
    }

    System.out.println(correct / x.rows());
    System.out.println(Arrays.mean(d.get(Arrays.range(5, x.rows())).asDouble()) / x.columns());
  }

  static double maxnot(DoubleArray array, int not) {
    Double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < array.size(); i++) {
      if (i == not) {
        continue;
      }
      double m = array.get(i);
      if (m > max) {
        max = m;
      }
    }
    return max;
  }


  private final Random random = new Random(123);

  private Vector shift(Vector record) {
    Vector.Builder builder = record.newBuilder();
    int start = random.nextInt(record.size() - 1);
    for (int i = start; i < record.size(); i++) {
      builder.add(record, i);
    }
    for (int i = 0; i < start; i++) {
      builder.add(record, i);
    }

    return builder.build();
  }

  @Test
  public void testShift() throws Exception {
    System.out.println(shift(Vector.of(1, 2, 3, 4, 5, 6)));
  }

  private DoubleArray computeOobAccuracy(RandomShapeletForest predictor, DataFrame x, Vector y) {

    BooleanArray oob = predictor.getOobIndicator();
    for (int i = 0; i < x.rows(); i++) {
      Vector record = x.loc().getRecord(i);
      List<Classifier> oobMembers = getOobMembers(oob.getRow(i), predictor.getEnsembleMembers());
      DoubleArray estimate = predictOob(oobMembers, record);

    }
    return null;
  }

  private List<Classifier> getOobMembers(BooleanArray oob, List<Classifier> classifiers) {
    List<Classifier> oobPredictors = new ArrayList<>();
    for (int i = 0; i < classifiers.size(); i++) {
      if (oob.get(i)) {
        oobPredictors.add(classifiers.get(i));
      }
    }

    return oobPredictors;
  }

  private DoubleArray predictOob(List<Classifier> members, Vector record) {

    return null;
  }

  public DataFrame vectorize(DataFrame x) {
    Set<Object> columns = new HashSet<>();
    for (Object recordKey : x.getIndex().keySet()) {
      List<Object> list = x.getRecord(recordKey).toList();
      columns.addAll(list.subList(1, list.size() - 1));
    }

    DataFrame.Builder builder = DataFrame.builder();
    builder.set("Class", x.get(0));
    for (Object column : columns) {
      if (StringUtils.isWhitespace(column.toString())) {
        continue;
      }
      Vector.Builder columnBuilder = Vector.Builder.of(Boolean.class);
      for (Object recordKey : x.getIndex().keySet()) {
        columnBuilder.add(x.getRecord(recordKey).toList().contains(column));
      }
      builder.set(column, columnBuilder);
    }

    return builder.build();
  }

  @Test
  public void testSequences() throws Exception {
    String ade = "G444";
    EntryReader in = new SequenceDatasetReader(
        new FileInputStream("/Users/isak-kar/Desktop/sequences/" + ade + ".seq"));

    DataFrame frame = new DataSeriesCollection.Builder(VectorType.STRING).readAll(in).build();
    frame = vectorize(frame);

    // System.out.println(frame.rows() + ", " + frame.columns());
    // Utils.setRandomSeed(32);
    frame = DataFrames.permuteRecords(frame);
    //
    Vector y = frame.get("Class");
    DataFrame x = frame.drop("Class");
    System.out.println(y.size());
    System.out.println(x.rows());
    Map<Object, Integer> freq = Vectors.count(y);
    int sum = freq.values().stream().reduce(0, Integer::sum);
    int min = freq.values().stream().min(Integer::min).get();
    System.out.println(freq + " => " + ((double) min / sum));
    //
    Predictor.Learner<? extends Classifier> forest =
        new RandomForest.Configurator(100).setMaximumFeatures(100).configure();
    // new RandomShapeletForest.Configurator(100).setF.configure();
    // new RandomShapeletForest.Configurator(25).setDistance(
    // new SlidingDistance(new HammingDistance())).configure();
    // new NearestNeighbours.Learner(1, new SimilarityDistance(
    // new SmithWatermanSimilarity(1, 0, 0)));

    Validator<Classifier> cv = crossValidator(10);
    cv.add(Evaluator.foldOutput(fold -> System.out.printf("Completed fold %d\n", fold)));
    Result result = cv.test(forest, x, y);
    System.out.println(result.getMeasures().mean());
    // System.out.println(result.getAverageConfusionMatrix().getPrecision("ade"));
    // System.out.println(result.getAverageConfusionMatrix().getRecall("ade"));
    // System.out.println(result.getAverageConfusionMatrix().getFMeasure("ade", 2));
    // System.out.println(result);
  }
}
