package org.mimirframework.examples;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.briljantframework.array.ArrayPrinter;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.LevelComparator;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.index.Index;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.DatasetReader;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.mimirframework.classification.RandomShapeletForest;
import org.mimirframework.classification.conformal.ClassifierCalibrator;
import org.mimirframework.classification.conformal.ConformalClassifier;
import org.mimirframework.classification.conformal.InductiveConformalClassifier;
import org.mimirframework.classification.conformal.ProbabilityCostFunction;
import org.mimirframework.classification.conformal.ProbabilityEstimateNonconformity;
import org.mimirframework.classification.conformal.evaluation.ConformalClassifierMeasure;
import org.mimirframework.evaluation.partition.Partition;
import org.mimirframework.evaluation.partition.SplitPartitioner;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class EarlyConformalPrediction {

  public static void main(String[] args) throws IOException {
    ArrayPrinter.setMinimumTruncateSize(10000);
    DataFrame data = DataFrames.permuteRecords(loadDatasetExample());
    DataFrame x = data.drop(0);
    Vector y = data.get(0);

    RandomShapeletForest.Configurator rsf = new RandomShapeletForest.Configurator(100);
    // BootstrapConformalClassifier.Learner pccl =
    // new BootstrapConformalClassifier.Learner(rsf.configure(), ProbabilityCostFunction.margin());

    ClassifierCalibrator calibrator = (nc, d, v) -> {
      Map<Integer, DoubleArray> lengthNc = new HashMap<>();
      int start = 5;
      for (int i = 0; i < d.columns() - start; i++) {
        lengthNc.put(i + start + 1, nc.estimate(d.loc().get(Range.of(0, i + start + 1)), v));
      }
      return (vector, o) -> lengthNc.get(vector.size());
    };

    ProbabilityEstimateNonconformity.Learner nc =
        // new DistanceNonconformity.Learner(1)
        new ProbabilityEstimateNonconformity.Learner(rsf.configure(),
            ProbabilityCostFunction.margin());

    InductiveConformalClassifier.Learner ccl =
        new InductiveConformalClassifier.Learner(nc, calibrator, false);
    //
    // System.out.println(ConformalClassifierValidator.crossValidator(10).test(pccl, x, y)
    // .getMeasures().groupBy("significance").collect(Vector::mean).sort(SortOrder.ASC));
    //
    //
    //
    //
    SplitPartitioner partitioner = new SplitPartitioner(0.2);
    Partition p = partitioner.partition(x, y).iterator().next();
    Partition p2 =
        partitioner.partition(p.getTrainingData(), p.getTrainingTarget()).iterator().next();
    //
    InductiveConformalClassifier cc = ccl.fit(p2.getTrainingData(), p2.getTrainingTarget());
    cc.calibrate(p2.getValidationData(), p2.getValidationTarget());
    System.out.println(p2.getTrainingData().rows());
    System.out.println(p2.getValidationData().rows());
    System.out.println(p.getTrainingData().rows());
    System.out.println(p.getValidationData().rows());
    // ConformalClassifier cc = pccl.fit(p.getTrainingData(), p.getTrainingTarget());
    evaluate(cc, p.getValidationData(), p.getValidationTarget());
  }

  private static void evaluate(ConformalClassifier pcc, DataFrame x, Vector y) {
    DoubleArray sign = DoubleArray.of(0.01, 0.05, 0.1);
    int start = 5;
    DoubleArray error = DoubleArray.zeros(x.columns() - start, sign.size());
    DoubleArray nClass = DoubleArray.zeros(x.columns() - start, sign.size());
    DataFrame.Builder result =
        DataFrame.builder(Integer.class, Double.class, Double.class, Double.class);
    result.setColumnIndex(Index.of("size", "significance", "error", "noClasses"));
    for (int i = 0; i < x.columns() - start; i++) {
      DataFrame prex = x.loc().get(Arrays.range(0, i + start + 1));
      DoubleArray estimate = pcc.estimate(prex);
      // System.out.println(prex);
      // System.out.println(estimate);

      for (int j = 0; j < sign.size(); j++) {
        ConformalClassifierMeasure measure =
            new ConformalClassifierMeasure(y, estimate, sign.get(j), pcc.getClasses());
        result.addRecord(Vector.of(i + start, sign.get(j), measure.getError(),
            measure.getNoClasses()));

        error.set(i, j, measure.getError());
        nClass.set(i, j, measure.getNoClasses());
      }
      // System.out.println(measure.getConfidence() + " " + measure.getCredibility());
      // if (measure.getConfidence() > 0.9 && measure.getCredibility() > 0.1) {
      // performance.set(i - 5, 0, measure.getAccuracy());
      // performance.set(i - 5, 1, measure.getNoClasses());
      // System.out.println(measure.getAccuracy() + ", " + measure.getNoClasses());
      // }
    }

    System.out.println(Arrays.mean(0, error));
    System.out.println(Arrays.mean(0, nClass));
    System.out.println(error);
    System.out.println(nClass);

    System.out.println(DataFrames.toString(
        result.build().groupBy("significance", "size").collect(Vector::mean)
            .sort(LevelComparator.of()), 10000));
  }

  public static DataFrame loadDatasetExample() throws IOException {
    // Dataset can be found here: http://www.cs.ucr.edu/~eamonn/time_series_data/
    String trainFile = "/Users/isak-kar/Downloads/dataset/Gun_Point/Gun_Point_TRAIN";
    String testFile = "/Users/isak-kar/Downloads/dataset/Gun_Point/Gun_Point_TEST";
    try (DatasetReader train = new MatlabDatasetReader(new FileInputStream(trainFile));
        DatasetReader test = new MatlabDatasetReader(new FileInputStream(testFile))) {
      DataFrame.Builder dataset = new DataSeriesCollection.Builder(double.class);
      dataset.readAll(train);
      dataset.readAll(test);
      return dataset.build();
    }
  }
}
