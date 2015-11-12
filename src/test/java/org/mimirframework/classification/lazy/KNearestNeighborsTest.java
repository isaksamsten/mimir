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

package org.mimirframework.classification.lazy;

import org.mimirframework.classification.NearestNeighbours;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.distance.DynamicTimeWarping;
import org.mimirframework.distance.EuclideanDistance;
import org.junit.Test;

public class KNearestNeighborsTest {

  @Test
  public void testClassifier() throws Exception {
    NearestNeighbours.Learner oneNearestNeighbours =
        new NearestNeighbours.Learner(1, new DynamicTimeWarping(EuclideanDistance.getInstance(), 3));

    DataFrame iris = DataFrames.dropIncompleteCases(Datasets.loadIris());
    Vector y = iris.get(4);
    DataFrame x = iris.drop(4);

//    Result res = ClassifierValidator.crossValidation(10).test(oneNearestNeighbours, x, y);
//    System.out.println(res);
    // ClassificationFrame train = DataSeriesInputStream.load(datasetPath + "TRAIN", 0,
    // Frame.FACTORY, DefaultTarget.FACTORY);
    // ClassificationFrame test = DataSeriesInputStream.load(datasetPath + "TEST", 0,
    // Frame.FACTORY, DefaultTarget.FACTORY);

    // System.out.println(train);
    // KNearestNeighbors.Classifier model = oneNearestNeighbours.fit(train);
    // Result result = Evaluators.holdOutValidation(oneNearestNeighbours, train, test);
    // System.out.println(result);
    //
    //
    // ZNormalizer<ClassificationFrame> normalizer = new ZNormalizer<>();
    // Transformer<ClassificationFrame> n = normalizer.fit(train);
    //
    // ClassificationFrame f = n.transform(train, ClassificationFrame.FACTORY);
    //
    //
    // Configurations<KNearestNeighbors> configurations =
    // Tuners.crossValidation(ClassificationFrame.FACTORY,
    // KNearestNeighbors.builder(), train, Comparator.naturalOrder(), 10,
    // Updaters.range("K", KNearestNeighbors.Builder::withNeighbors, 1, 10, 1)
    // );
    //
    // System.out.println(configurations);
    //
    //
    // ShapeletTree.Builder timeSeriesTree = ShapeletTree.withSplitter(RandomShapeletSplitter
    // .withDistance(EarlyAbandonSlidingDistance.create(Distance.EUCLIDEAN))
    // .withLowerLength(2)
    // .withInspectedShapelets(-1)
    // .withUpperLength(-1));
    //
    // Ensemble<Frame> ensemble = Ensemble.withMember(timeSeriesTree)
    // .withSampler(Bootstrap.create())
    // .withSize(100)
    // .create();
    //
    // long start = System.currentTimeMillis();
    // Ensemble.Classifier<Frame> model2 = ensemble.fit(train);
    // System.out.println("Fit: " + ((System.currentTimeMillis() - start) / 1000) + " seconds");
    //
    //
    // Ensemble<Dataset> ensemble1 = Ensemble.withMember(DecisionTree.withSplitter(RandomSplitter
    // .withMaximumFeatures(2))).setRandomizer(Bootstrap.create()).create();
    // Ensemble.Classifier<Dataset> model2 = ensemble1.fit(train);
    // start = System.currentTimeMillis();
    // Predictions predictions = model2.predict(test.getDataset());
    // System.out.println("Predict: " + ((System.currentTimeMillis() - start) / 1000) + " seconds");
    // Target testTarget = test.getTarget();
    // int correct = 0;
    // for (int i = 0; i < predictions.size(); i++) {
    // Prediction p = predictions.get(i);
    // if (p.getValue().equals(testTarget.getValue(i))) {
    // correct += 1;
    // } else {
    // System.out.printf("%s for %s (real %s)%n", p.getPredictedProbability(), p.getValue(),
    // testTarget.getValue(i));
    // }
    // }
    // System.out.printf("Error: %.5f ", (1 - ((double) correct) / test.rows()));

  }
}
