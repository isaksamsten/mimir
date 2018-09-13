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
package org.briljantframework.mimir.shapelet;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.classification.tree.Gain;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.distance.EarlyAbandonSlidingDistance;
import org.briljantframework.mimir.timeseries.data.TimeSeries;

import com.carrotsearch.hppc.ObjectDoubleMap;
import com.carrotsearch.hppc.ObjectDoubleOpenHashMap;

/**
 * Created by isak on 2017-06-13.
 */
public class TfIdfShapeletTransform {


  private final Iterable<ShapeletWithIndex> it;
  private final double maxEuclideanDist;
  private final int minDf, maxDf;

  public TfIdfShapeletTransform(Iterable<ShapeletWithIndex> it, double maxEuclideanDist, int minDf,
      int maxDf) {
    this.it = Objects.requireNonNull(it);
    this.maxEuclideanDist = maxEuclideanDist;
    this.minDf = minDf;
    this.maxDf = maxDf;
  }

  public List<TfIdfShapeletResult> transform(Input<TimeSeries> input, List<?> labels) {
    ClassSet classSet = new ClassSet(labels, Array.copyOf(new HashSet<>(labels)));

    ShapeletCounter counter = new ShapeletCounter(maxEuclideanDist);
    List<TfIdfShapeletResult> results = Collections.synchronizedList(new ArrayList<>());
    ExecutorService executorService =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());



    // Iterable<ShapeletWithIndex> it = new AllShapeletIterable(input, min, max);
    // Iterable<ShapeletWithIndex> it = new RandomShapeletIterable(1000, input, 0.025, 1);
    int no = 0;
    for (ShapeletWithIndex shapeletWithIndex : it) {
      executorService.execute(new TfIdfShapeletComputation(classSet, no++,
          shapeletWithIndex.getShapelet(), shapeletWithIndex.getIndex(), input, labels, minDf,
          maxDf, counter, results, !Double.isNaN(maxEuclideanDist)));
    }
    executorService.shutdown();
    try {
      executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
    } catch (InterruptedException e) {
      throw new RuntimeException("Executor took to long to finish.", e);
    }

    return results;
  }

  public static Threshold bestDistanceThresholdInSample(ClassSet classSet, Input<TimeSeries> x,
      List<?> y, Shapelet shapelet) {
    double sum = 0.0;
    List<ExampleDistance> distances = new ArrayList<>();
    for (Example example : classSet) {
      TimeSeries record = x.get(example.getIndex());
      double distance = EarlyAbandonSlidingDistance.getInstance().compute(record, shapelet);
      distances.add(new ExampleDistance(distance, example));
      if (!Is.NA(distance) && !Double.isInfinite(distance)) {
        sum += distance;
      }
    }

    Collections.sort(distances);
    return findBestThreshold(distances, classSet, y, sum);
  }

  public static Threshold findBestThreshold(List<ExampleDistance> distances, ClassSet classSet,
      List<?> y, double distanceSum) {
    ObjectDoubleMap<Object> lt = new ObjectDoubleOpenHashMap<>();
    ObjectDoubleMap<Object> gt = new ObjectDoubleOpenHashMap<>();

    List<Object> presentTargets = classSet.getTargets();
    DoubleArray ltRelativeFrequency = DoubleArray.zeros(presentTargets.size());
    DoubleArray gtRelativeFrequency = DoubleArray.zeros(presentTargets.size());

    double ltWeight = 0.0, gtWeight = 0.0;

    // Initialize all value to the right (i.e. all values are larger than the initial threshold)
    for (ClassSet.Sample sample : classSet.samples()) {
      double weight = sample.getWeight();
      gtWeight += weight;

      lt.put(sample.getTarget(), 0);
      gt.put(sample.getTarget(), weight);
    }
    ExampleDistance ed = distances.get(0);

    // Transfer weights from the initial example
    Example first = ed.example;
    Object prevTarget = y.get(first.getIndex());
    gt.addTo(prevTarget, -first.getWeight());
    lt.addTo(prevTarget, first.getWeight());
    gtWeight -= first.getWeight();
    ltWeight += first.getWeight();

    double prevDistance = ed.distance;
    double lowestImpurity = Double.POSITIVE_INFINITY;
    double threshold = Double.isFinite(ed.distance) ? ed.distance / 2 : 0;
    Gain gain = Gain.INFO;
    double ltGap = 0.0, gtGap = distanceSum, largestGap = Double.NEGATIVE_INFINITY;
    for (int i = 1; i < distances.size(); i++) {
      ed = distances.get(i);
      Object target = y.get(ed.example.getIndex());

      // IF previous target NOT EQUALS current target and the previous distance equals the current
      // (except for the first)
      boolean notSameDistance = ed.distance != prevDistance;
      boolean firstOrEqualTarget = prevTarget == null || !prevTarget.equals(target);
      boolean firstIteration = i == 1;
      if (firstIteration || notSameDistance && firstOrEqualTarget) {

        // Generate the relative frequency distribution
        for (int j = 0; j < presentTargets.size(); j++) {
          Object presentTarget = presentTargets.get(j);
          ltRelativeFrequency.set(j, ltWeight != 0 ? lt.get(presentTarget) / ltWeight : 0);
          gtRelativeFrequency.set(j, gtWeight != 0 ? gt.get(presentTarget) / gtWeight : 0);
        }

        // If this split is better, update the threshold
        double impurity =
            gain.compute(ltWeight, ltRelativeFrequency, gtWeight, gtRelativeFrequency);
        double gap = (1 / ltWeight * ltGap) - (1 / gtWeight * gtGap);
        boolean lowerImpurity = impurity < lowestImpurity;
        boolean equalImpuritySmallerGap = impurity == lowestImpurity && gap > largestGap;
        if (lowerImpurity || equalImpuritySmallerGap) {
          lowestImpurity = impurity;
          largestGap = gap;
          threshold =
              Double.isFinite(ed.distance) ? (ed.distance + prevDistance) / 2 : prevDistance;
        }
      }

      /*
       * Move cursor one example forward, and adjust the weights accordingly. Then calculate the new
       * gain for moving the threshold. If this results in a cleaner split, adjust the threshold (by
       * taking the average of the current and the previous value).
       */
      double weight = ed.example.getWeight();
      ltWeight += weight;
      gtWeight -= weight;
      lt.addTo(target, weight);
      gt.addTo(target, -weight);

      if (Double.isFinite(ed.distance)) {
        ltGap += ed.distance;
        gtGap -= ed.distance;
        prevDistance = ed.distance;
      }

      prevTarget = target;
    }

    double minimumMargin = Double.POSITIVE_INFINITY;
    return new Threshold(threshold, gain.getImpurity().impurity(classSet) - lowestImpurity,
        largestGap, minimumMargin);
  }


  private static class TfIdfShapeletComputation implements Runnable {

    private final ClassSet classSet;
    private final List<TfIdfShapeletResult> resultList;
    private final Shapelet shapelet;
    private final Input<TimeSeries> input;
    private final int minDf, maxDf;
    private final ShapeletCounter counter;
    private final int timeSeriesIndex;
    private final int no;
    private final List<?> output;
    private final boolean fixedDistance;

    private TfIdfShapeletComputation(ClassSet classSet, int no, Shapelet s, int i,
        Input<TimeSeries> input, List<?> output, int minDf, int maxDf, ShapeletCounter counter,
        List<TfIdfShapeletResult> resultList, boolean fixedDistance) {
      this.output = output;
      this.classSet = classSet;
      this.resultList = resultList;
      this.shapelet = s;
      this.input = input;
      this.minDf = minDf;
      this.maxDf = maxDf;
      this.counter = counter;
      this.timeSeriesIndex = i;
      this.no = no;
      this.fixedDistance = fixedDistance;
    }

    @Override
    public void run() {
      Threshold threshold = bestDistanceThresholdInSample(classSet, input, output, shapelet);
      DoubleArray tf = DoubleArray.zeros(input.size());
      List<ShapeletMatch> matches = new ArrayList<>();
      double noDocs = 0;
      double thresholdValue = Math.sqrt(threshold.getThreshold() * threshold.getThreshold() * shapelet.size());
      for (int l = 0; l < input.size(); l++) {
        ShapeletMatch value;
        if (fixedDistance) {
          value = counter.numberOfMatches(input.get(l), shapelet);
        } else {
          value = counter.numberOfMatches(input.get(l), shapelet,
              thresholdValue);
        }

        tf.set(l, value.getCount());
        if (value.getCount() > 0) {
          noDocs++;
        }
        matches.add(value);
      }
      if (noDocs <= maxDf && noDocs >= minDf) {
        double log = Math.log(input.size() / noDocs);

        DoubleArray scores = Arrays.times(tf, log);
        TfIdfShapeletResult result =
            new TfIdfShapeletResult(no, shapelet.start(), shapelet.size(), timeSeriesIndex,
                threshold.getInformationGain(), thresholdValue, scores, matches);
        // System.err.println("Added: " + no + " df=" + noDocs);
        resultList.add(result);
      }
    }
  }


  private static class Threshold {
    private final double threshold, lowestImpurity, largestGap, minimumMargin;

    public Threshold(double threshold, double lowestImpurity, double largestGap,
        double minimumMargin) {

      this.threshold = threshold;
      this.lowestImpurity = lowestImpurity;
      this.largestGap = largestGap;
      this.minimumMargin = minimumMargin;
    }

    public double getThreshold() {
      return threshold;
    }

    public double getInformationGain() {
      return lowestImpurity;
    }

    public double getLargestGap() {
      return largestGap;
    }

    public double getMinimumMargin() {
      return minimumMargin;
    }

    @Override
    public String toString() {
      return "Threshold{" + "threshold=" + threshold + ", lowestImpurity=" + lowestImpurity
          + ", largestGap=" + largestGap + ", minimumMargin=" + minimumMargin + '}';
    }
  }

  private static class ExampleDistance implements Comparable<ExampleDistance> {
    private final double distance;
    private final Example example;

    public ExampleDistance(double distance, Example example) {
      this.example = example;
      this.distance = distance;
    }

    public double getDistance() {
      return distance;
    }

    public Example getExample() {
      return example;
    }

    @Override
    public int compareTo(ExampleDistance o) {
      return Double.compare(distance, o.distance);
    }
  }
}
