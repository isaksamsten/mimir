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

import java.io.FileInputStream;
import java.io.IOException;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.DatasetReader;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.briljantframework.mimir.ArrayOutput;
import org.briljantframework.mimir.Inputs;
import org.briljantframework.mimir.Instance;
import org.briljantframework.mimir.Outputs;
import org.briljantframework.mimir.classification.conformal.ClassifierCalibrator;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ProbabilityCostFunction;
import org.briljantframework.mimir.classification.conformal.ProbabilityEstimateNonconformity;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.junit.Test;

/**
 * Created by isak on 11/16/15.
 */
public class LearnerTest {

  @Test
  public void testTesda2() throws Exception {
    // ArrayPrinter.setMinimumTruncateSize(100000);
    // DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    // DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    // Vector y = iris.get("Class");
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
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());

    // Remove the class variable from the input data and set each NA value to the column mean
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));

    // Get the class variable
    Vector y = iris.get("Class");


    // Create a classifier learner to use for estimating the non-conformity scores
    RandomForest.Learner classifier = new RandomForest.Learner(100);
    ClassifierValidator<Instance, RandomForest> rfv = ClassifierValidator.crossValidator(10);

    System.out.println(
        rfv.test(new RandomForest.Learner(100), Inputs.newInput(x), new ArrayOutput<>(y.toList()))
            .getMeasures().mean());

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

    Result<?> result = validator.test(cp, Inputs.newInput(x), Outputs.newOutput(y));

    // Get the measures
    DataFrame measures = result.getMeasures();

    // Compute the mean of all measures grouped by significance level
    DataFrame meanPerSignificance =
        measures.groupBy(Double.class, v -> String.format("%.2f", v), "significance")
            .collect(Vector::mean).sort(SortOrder.ASC);
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
      DataFrame.Builder dataset = new DataSeriesCollection.Builder(double.class);
      dataset.readAll(train);
      dataset.readAll(test);
      return dataset.build();
    }
  }
}
