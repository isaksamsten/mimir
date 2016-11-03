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
/**
 * This copy of Woodstox XML processor is licensed under the
 * Apache (Software) License, version 2.0 ("the License").
 * See the License for details about distribution rights, and the
 * specific rights regarding derivate works.
 *
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/
 *
 * A copy is also included in the downloadable source code package
 * containing Woodstox, in file "ASL2.0", under the same directory
 * as this file.
 */
package org.briljantframework.examples;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
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
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CombinedDomainXYPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.briljantframework.mimir.classification.conformal.ClassifierCalibrator;
import org.briljantframework.mimir.classification.conformal.ClassifierNonconformity;
import org.briljantframework.mimir.classification.conformal.ConformalClassifier;
import org.briljantframework.mimir.classification.conformal.DistanceNonconformity;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierMeasure;
import org.briljantframework.mimir.evaluation.partition.Partition;
import org.briljantframework.mimir.evaluation.partition.SplitPartitioner;
import org.briljantframework.mimir.jfree.ArrayYDataset;
import org.briljantframework.mimir.jfree.Plots;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class EarlyConformalPrediction {

  public static void main(String[] args) throws IOException {
    ArrayPrinter.setMinimumTruncateSize(10000);
    DataFrame data = DataFrames.permuteRecords(loadDatasetExample());
    DataFrame x = data.drop(0);
    Vector y = data.get(0);

    // RandomShapeletForest.Configurator rsf = new RandomShapeletForest.Configurator(100);
    // BootstrapConformalClassifier.Learner pccl =
    // new BootstrapConformalClassifier.Learner(rsf.configure(), ProbabilityCostFunction.margin());

    ClassifierCalibrator calibrator = (nc, d, v) -> {
      Map<Integer, DoubleArray> lengthNc = new HashMap<>();
      int start = 5;
      for (int i = 0; i < d.columns() - start; i++) {
        DataFrame x1 = d.values().get(Range.of(0, i + start + 1));
        System.out.println(x1.columns() + " => " + (i + start + 1));
        lengthNc.put(i + start + 1, nc.estimate(x1, v));
      }
      return (vector, o) -> lengthNc.get(vector.size());
    };
    //
    ClassifierNonconformity.Learner nc = new DistanceNonconformity.Learner(1);
    // new ProbabilityEstimateNonconformity.Learner(rsf.configure(),
    // ProbabilityCostFunction.margin());
    //
    InductiveConformalClassifier.Learner ccl =
        new InductiveConformalClassifier.Learner(nc, calibrator, false);
    //
    // System.out.println(ConformalClassifierValidator.crossValidator(10).test(pccl, x, y)
    // .getMeasures().groupBy("significance").collect(Vector::mean).sort(SortOrder.ASC));
    //
    //
    //
    //
    SplitPartitioner partitioner = new SplitPartitioner(0.35);
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

  private static void evaluate(ConformalClassifier cc, DataFrame x, Vector y) {
    DoubleArray sign = DoubleArray.of(0.01, 0.05, 0.1);
    int start = 5;
    DoubleArray error = DoubleArray.zeros(x.columns() - start, sign.size());
    DoubleArray nClass = DoubleArray.zeros(x.columns() - start, sign.size());
    DoubleArray meanPvalue = DoubleArray.zeros(x.columns() - start, sign.size());
    DataFrame.Builder result =
        DataFrame.builder(Integer.class, Double.class, Double.class, Double.class);
    result.setColumnIndex(Index.of("size", "significance", "error", "noClasses"));
    for (int i = 0; i < x.columns() - start; i++) {
      DataFrame prex = x.values().get(Arrays.range(0, i + start + 1));
      DoubleArray estimate = cc.estimate(prex);
      System.out.printf("Processing %d/%d\n", i, x.columns() - start);

      for (int j = 0; j < sign.size(); j++) {
        ConformalClassifierMeasure measure =
            new ConformalClassifierMeasure(y, estimate, sign.get(j), cc.getClasses());
        result.addRecord(Vector.of(i + start + 1, sign.get(j), measure.getError(),
            measure.getNoClasses()));

        error.set(i, j, measure.getError());
        nClass.set(i, j, measure.getNoClasses());
        meanPvalue.set(i, j, measure.getAveragePvalue());
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
    System.out.println(meanPvalue);

    List<String> seriesKeys = sign.mapToObj(String::valueOf).toList();
    XYDataset errorDataset = new ArrayYDataset(error.boxed(), seriesKeys);
    XYDataset nClassDataset = new ArrayYDataset(nClass.boxed(), seriesKeys);

    XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(true, false);
    NumberAxis rangeAxis = new NumberAxis();
    NumberAxis domainAxis = new NumberAxis();
    XYPlot errorPlot = new XYPlot(errorDataset, domainAxis, rangeAxis, renderer);
    rangeAxis.setLowerBound(0);
    rangeAxis.setUpperBound(0.3);

    XYPlot nClassPlot = new XYPlot(nClassDataset, domainAxis, new NumberAxis(), renderer);
    CombinedDomainXYPlot cplot = new CombinedDomainXYPlot(new NumberAxis()) {
      @Override
      public LegendItemCollection getLegendItems() {
        return ((XYPlot) getSubplots().get(0)).getLegendItems();
      }
    };
    cplot.add(errorPlot);
    cplot.add(nClassPlot);
    Plots.show(cplot);

    System.out.println(DataFrames.toString(
        result.build().groupBy("significance", "size").collect(Vector::mean)
            .sort(LevelComparator.of()), 10000));
  }

  public static DataFrame loadDatasetExample() throws IOException {
    // Dataset can be found here: http://www.cs.ucr.edu/~eamonn/time_series_data/
    String name = "ECGFiveDays";
    String trainFile = String.format("/Users/isak-kar/Downloads/dataset2/%s/%s_TRAIN", name, name);
    String testFile = String.format("/Users/isak-kar/Downloads/dataset2/%s/%s_TEST", name, name);
    try (DatasetReader train = new MatlabDatasetReader(new FileInputStream(trainFile));
        DatasetReader test = new MatlabDatasetReader(new FileInputStream(testFile))) {
      DataFrame.Builder dataset = new DataSeriesCollection.Builder(double.class);
      dataset.readAll(train);
      dataset.readAll(test);
      return dataset.build();
    }
  }
}
