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

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.DatasetReader;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.EnsembleEvaluator;
import org.briljantframework.mimir.classification.RandomShapeletForest;
import org.briljantframework.mimir.classification.ShapeletTree;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;

/**
 * Created by isak on 11/13/15.
 */
public class RandomShapeletForestExamples {

  public static void main(String[] args) throws IOException {
    // This is a dataset which we load
    DataFrame data = DataFrames.permuteRecords(Datasets.loadSyntheticControl());

    // See: loadDatasetExample() for an example of how to load a dataset
    // DataFrame data = DataFrames.permuteRecords(loadDatasetExample());

    // The first column of the dataset contains the class, so we drop it and then extract it
    DataFrame x = data.drop(0);
    Vector y = data.get(0);

    Validator<RandomShapeletForest> cv = ClassifierValidator.crossValidator(10);
    cv.add(Evaluator.foldOutput(i -> System.out.printf("Fold: %d\n", i)));
    cv.add(EnsembleEvaluator.INSTANCE);

    // Initialize a random shapelet forest configurator; 100 trees
    RandomShapeletForest.Configurator config = new RandomShapeletForest.Configurator(100);

    // Use information gain
    config.setAssessment(ShapeletTree.Learner.Assessment.IG);

    // The minimum shapelet length is 2.5% of the time series length
    config.setLowerLength(0.025);
    config.setUpperLength(1.0);

    // Sample 10 shapelets at each node
    config.setMaximumShapelets(10);
    RandomShapeletForest.Learner forest = config.configure();

    // Evaluate the classifier
    Result result = cv.test(forest, x, y);

    // Note that precision and recall is not implemented yet
    System.out.println("Results averaged over 10-fold cross-validation");
    DataFrame measures = result.getMeasures();
    System.out.println(measures.mean());
  }

  public static DataFrame loadDatasetExample() throws IOException {
    // Dataset can be found here: http://www.cs.ucr.edu/~eamonn/time_series_data/
    String trainFile = "/home/isak/Projects/datasets/dataset/Gun_Point/Gun_Point_TRAIN";
    String testFile = "/home/isak/Projects/datasets/dataset/Gun_Point/Gun_Point_TEST";
    try (DatasetReader train = new MatlabDatasetReader(new FileInputStream(trainFile));
        DatasetReader test = new MatlabDatasetReader(new FileInputStream(testFile))) {
      DataFrame.Builder dataset = new DataSeriesCollection.Builder(double.class);
      dataset.readAll(train);
      dataset.readAll(test);
      return dataset.build();
    }
  }

}
