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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.apache.commons.math3.util.Pair;
import org.briljantframework.DoubleVector;
import org.briljantframework.data.Collectors;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.classification.tree.Entropy;
import org.briljantframework.mimir.classification.tree.TreeBranch;
import org.briljantframework.mimir.classification.tree.TreeNode;
import org.briljantframework.mimir.classification.tree.pattern.*;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.distance.EuclideanDistance;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.shapelet.*;
import org.briljantframework.mimir.timeseries.data.MultivariateTimeSeries;
import org.briljantframework.mimir.timeseries.data.MultivariateTimeSeriesSchema;
import org.briljantframework.mimir.timeseries.data.TimeSeries;
import org.junit.Test;

/**
 * Created by isak on 2017-03-27.
 */
public class RandomShapeletForestTest {


  @Test
  public void testErrorRate() throws Exception {
    Pair<Input<MultivariateTimeSeries>, List<Object>> i =
        readData("/Users/isak/Projects/dataset/synthetic_control/synthetic_control_TRAIN");

    Input<MultivariateTimeSeries> in = i.getFirst();
    List<Object> out = i.getSecond();

    ClassifierValidator<MultivariateTimeSeries, Object> cv =
        ClassifierValidator.crossValidator(10);

//     RandomPatternForest.Learner<MultivariateTimeSeries, Object> learner =
//     getChangePointForest(100);
    RandomShapeletForest.Learner<Object> learner = new RandomShapeletForest.Learner<>();
    learner.set(Ensemble.SIZE, 100);
//    RandomShapeletForest<Object> fit = learner.fit(in, out);
//
//    ShapeletStorage shapeletStorage = new ShapeletStorage(EuclideanDistance.getInstance(), 15);
//    importance(fit.getMembers(), shapeletStorage, in.size());
//
//    for (ShapeletScore shapeletScore : shapeletStorage.getSorted()) {
//      System.out.println(shapeletScore);
//    }



//    StatisticalSummary collect = in.stream().map(t -> t.getDimension(0))
//        .map(t -> t.getChangePoints().length).collect(Collectors.statisticalSummary());
//    System.out.println(collect);

    Result<Object> test = cv.test(learner, in, out);
    System.out.println(test.getMeasures().reduce(Series::mean));
    System.out.println(test.getFitTime());
    System.out.println(test.getPredictTime());
  }

  private void importance(List<ProbabilityEstimator<MultivariateTimeSeries, Object>> members,
      ShapeletStorage shapeletStorage, double noExamples) {
    for (ProbabilityEstimator<MultivariateTimeSeries, Object> member : members) {
      PatternTree<MultivariateTimeSeries, Object> pt =
          (PatternTree<MultivariateTimeSeries, Object>) member;
      updateImportance(pt.getRootNode(), shapeletStorage, noExamples);
    }
  }

  private void updateImportance(TreeNode<MultivariateTimeSeries> rootNode,
      ShapeletStorage shapeletStorage, double noExamples) {
    if (rootNode instanceof TreeBranch) {
      DistanceTest<MultivariateTimeSeries, MultivariateShapelet> treeNodeTest =
          (DistanceTest<MultivariateTimeSeries, MultivariateShapelet>) ((TreeBranch<MultivariateTimeSeries>) rootNode)
              .getTreeNodeTest();


      double imp = Entropy.getInstance().impurity(((TreeBranch) rootNode).getClassSet());
      double score = rootNode.getWeight() * (imp - ((TreeBranch) rootNode).getImpurity());
      shapeletStorage.add(treeNodeTest.getShapelet().getShapelet(), score);
      updateImportance(((TreeBranch<MultivariateTimeSeries>) rootNode).getLeft(), shapeletStorage,
          noExamples);
      updateImportance(((TreeBranch<MultivariateTimeSeries>) rootNode).getRight(), shapeletStorage,
          noExamples);
    }


  }

  private RandomPatternForest.Learner<MultivariateTimeSeries, Object> getChangePointForest(
      int size) {

    ShapeletCounter counter = new ShapeletCounter(0.1);
    return new RandomPatternForest.Learner<>(
        new SamplingPatternFactory<MultivariateTimeSeries, MultivariateShapelet>() {
          @Override
          protected MultivariateShapelet createPattern(MultivariateTimeSeries input) {
            int dim = ThreadLocalRandom.current().nextInt(input.dimensions());
            TimeSeries uts = input.getDimension(dim);
//            int[] changePoints = uts.getChangePoints();
            int timeSeriesLength = uts.size();
            int lower = (int) Math.round(timeSeriesLength * 0.025);
            int upper = (int) Math.round(timeSeriesLength * 1.0);
            if (lower < 2) {
              lower = 2;
            }

            if (Math.addExact(upper, lower) > timeSeriesLength) {
              upper = timeSeriesLength - lower;
            }
            if (lower == upper) {
              upper -= 2;
            }
            if (upper < 1) {
              return null;
            }

            // int start = changePoints[ThreadLocalRandom.current().nextInt(changePoints.length)];
            // int maxLen = timeSeriesLength - start;
            // int length =
            // maxLen <= lower ? maxLen : ThreadLocalRandom.current().nextInt(lower, maxLen);
            int length = ThreadLocalRandom.current().nextInt(upper) + lower;
            int start = ThreadLocalRandom.current().nextInt(timeSeriesLength - length);

            // System.out.println("uts: "+timeSeriesLength+" start: "+ start + " maxLen: " + maxLen
            // + " lower: " + lower);
            return new MultivariateShapelet(dim,
                new IndexSortedNormalizedShapelet(start, length, uts));
          }
        }, new PatternDistance<MultivariateTimeSeries, MultivariateShapelet>() {
          @Override
          public double computeDistance(MultivariateTimeSeries mts, MultivariateShapelet mtsS) {
            ShapeletMatch v = counter.numberOfMatches(mts.getDimension(mtsS.getDimension()), mtsS.getShapelet());
//            System.out.println("Matches: " + v);
            return v.getCount();
          }
        }, size);
  }

  private static Pair<Input<MultivariateTimeSeries>, List<Object>> readData(String filePath)
      throws IOException {
    // Construct the input and output variables
    MultivariateTimeSeriesSchema schema = new MultivariateTimeSeriesSchema(1);
    Input<MultivariateTimeSeries> input = schema.newInput();
    List<Object> output = new ArrayList<>();

    // Read the file
    List<String> data = Files.readAllLines(Paths.get(filePath));
    Collections.shuffle(data, ThreadLocalRandom.current());
    for (String line : data) {
      String[] split = line.trim().split("\\s+");
      output.add(Double.parseDouble(split[0]));

      TimeSeries timeSeries = getTimeSeries(1, split);
      input.add(new MultivariateTimeSeries(timeSeries));
    }
    return new Pair<>(input, output);
  }

  private static TimeSeries getTimeSeries(int start, String[] split) {
    double[] ts = new double[split.length - start];
    for (int i = start; i < split.length; i++) {
      ts[i - start] = Double.parseDouble(split[i]);
    }
    // adding the same dimension twice - as an example. Each time-series should be distinct here.
    return TimeSeries.normalizedCopyOf(Series.copyOf(ts));
  }
}
