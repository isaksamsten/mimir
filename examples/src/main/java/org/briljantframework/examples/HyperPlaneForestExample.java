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

import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.MixedDataFrame;
import org.briljantframework.data.dataframe.transform.ZNormalizer;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.RdsDatasetReader;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.Ensemble;
import org.briljantframework.mimir.classification.EnsembleEvaluator;
import org.briljantframework.mimir.classification.HyperPlaneTree;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.jfree.Plots;
import org.jfree.chart.plot.XYPlot;

/**
 * @author Isak Karlsson
 */
public class HyperPlaneForestExample {

  public static void main(String[] args) throws IOException {

    // DataFrame data = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame data =
        Datasets.load(MixedDataFrame.Builder::new, new RdsDatasetReader(new FileInputStream(
            "/Users/isak-kar/Desktop/breast-cancer-wisconsin.txt")));
    String classCol = "Class";
    DataFrame x =
        new ZNormalizer().fitTransform(data.drop(classCol).apply(
            v -> v.set(v.where(Is::NA), v.mean())));
    Vector y = data.get(classCol);
    System.out.println(x);
    System.out.println(y);
    Validator<RandomForest> cv = ClassifierValidator.crossValidator(10);
    cv.add(EnsembleEvaluator.INSTANCE);
    cv.add(Evaluator.foldOutput(i -> System.out.format("%d ", i)));
    Ensemble.BaseLearner<Classifier> hyperTree =
        (classSet, classes) -> new HyperPlaneTree.Learner(classSet, classes, 10);

    IntArray trees = IntArray.of(1, 10, 100, 1000, 10000);
    DoubleArray error = DoubleArray.zeros(trees.size(), 2);
    for (int i = 0; i < trees.size(); i++) {
      int tree = trees.get(i);
      System.out.printf("Running %d trees\n", tree);
      RandomForest.Configurator configurator = new RandomForest.Configurator(tree);
      configurator.setBaseLearner(hyperTree);
      RandomForest.Learner hpRf = configurator.configure();
      RandomForest.Learner rf = configurator.setBaseLearner(null).configure();

      System.out.println("Running hyper forest ... ");
      Result hpR = cv.test(hpRf, x, y);
      System.out.println("Running random forest ... ");
      Result rfR = cv.test(rf, x, y);
      Vector meanHpR = hpR.getMeasures().mean();
      Vector meanRfR = rfR.getMeasures().mean();
      error.set(i, 0, 1 - meanHpR.getAsDouble("accuracy"));
      error.set(i, 1, 1 - meanRfR.getAsDouble("accuracy"));
    }

    XYPlot line = Plots.line(error);
    line.getRangeAxis().setLowerBound(0);
    line.getRangeAxis().setUpperBound(0.1);
    Plots.show(line);
  }
}
