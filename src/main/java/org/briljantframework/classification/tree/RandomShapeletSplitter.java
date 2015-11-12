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

package org.briljantframework.classification.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.distance.Distance;
import org.briljantframework.shapelet.IndexSortedNormalizedShapelet;
import org.briljantframework.shapelet.Shapelet;

import com.carrotsearch.hppc.IntDoubleMap;
import com.carrotsearch.hppc.IntDoubleOpenHashMap;
import com.carrotsearch.hppc.ObjectDoubleMap;
import com.carrotsearch.hppc.ObjectDoubleOpenHashMap;

/**
 * @author Isak Karlsson
 */
public class RandomShapeletSplitter extends ShapeletSplitter {
  protected final Random random = new Random();
  private final int inspectedShapelets;

  private final int lowerLength;
  private final int upperLength;
  private final double alpha;



  /**
   * Instantiates a new Random shapelet splitter.
   *
   * @param builder the builder
   */
  public RandomShapeletSplitter(Builder builder) {
    super(builder.distance, builder.gain);
    this.inspectedShapelets = builder.shapelets;
    this.upperLength = builder.upper;
    this.lowerLength = builder.lower;
    this.alpha = builder.alpha;
  }

  public static Builder withDistance(Distance distance) {
    return new Builder(distance);
  }

  public int getInspectedShapelets() {
    return inspectedShapelets;
  }

  public int getLowerLength() {
    return lowerLength;
  }

  public int getUpperLength() {
    return upperLength;
  }

  public double getAlpha() {
    return alpha;
  }

  public TreeSplit<ShapeletThreshold> find(ClassSet classSet, DataFrame x, Vector y) {
    int timeSeriesLength = x.columns();
    int upper = this.upperLength;
    int lower = this.lowerLength;

    if (upper < 0) {
      upper = timeSeriesLength;
    }
    if (lower < 2) {
      lower = 2;
    }

    if (Math.addExact(upper, lower) > timeSeriesLength) {
      upper = timeSeriesLength - lower;
    }

    int maxShapelets = this.inspectedShapelets;
    if (maxShapelets < 0) {
      int length = upper - lower;
      maxShapelets = (int) Math.round(Math.sqrt((length * (length + 1) / 2)));
    }
    List<Shapelet> shapelets = new ArrayList<>(maxShapelets);
    for (int i = 0; i < maxShapelets; i++) {
      Vector timeSeries =
          x.loc().getRecord(classSet.getRandomSample().getRandomExample().getIndex());
      int length = random.nextInt(upper) + lower;
      int start = random.nextInt(timeSeriesLength - length);
      shapelets.add(new IndexSortedNormalizedShapelet(start, length, timeSeries));
    }

    return findBestSplit(classSet, x, y, shapelets);
  }

  protected TreeSplit<ShapeletThreshold> findBestSplit(ClassSet classSet, DataFrame x, Vector y,
      List<Shapelet> shapelets) {
    TreeSplit<ShapeletThreshold> bestSplit = null;
    Threshold bestThreshold = Threshold.inf();
    for (Shapelet shapelet : shapelets) {
      IntDoubleMap distanceMap = new IntDoubleOpenHashMap();
      Threshold threshold = bestDistanceThresholdInSample(classSet, x, y, shapelet, distanceMap);
      boolean lowerImpurity = threshold.impurity < bestThreshold.impurity;
      boolean equalImpuritySmallerGap =
          threshold.impurity == bestThreshold.impurity && threshold.gap > bestThreshold.gap;
      if (lowerImpurity || equalImpuritySmallerGap) {
        bestSplit = split(distanceMap, classSet, threshold.threshold, shapelet);
        bestThreshold = threshold;
      }
    }

    if (bestSplit != null) {
      bestSplit.setImpurity(bestThreshold.impurity);
    }
    return bestSplit;
  }

  protected Threshold bestDistanceThresholdInSample(ClassSet classSet, DataFrame x, Vector y,
      Shapelet shapelet, IntDoubleMap memoizedDistances) {

    double sum = 0.0;
    List<ExampleDistance> distances = new ArrayList<>();
    Distance distanceMetric = getDistanceMetric();
    for (Example example : classSet) {
      double distance = distanceMetric.compute(x.loc().getRecord(example.getIndex()), shapelet);
      memoizedDistances.put(example.getIndex(), distance);
      distances.add(new ExampleDistance(distance, example));
      sum += distance;
    }

    Collections.sort(distances);
    return findBestThreshold(distances, classSet, y, sum);
  }

  public Threshold findBestThreshold(List<ExampleDistance> distances, ClassSet classSet, Vector y,
      double distanceSum) {
    ObjectDoubleMap<Object> lt = new ObjectDoubleOpenHashMap<>();
    ObjectDoubleMap<Object> gt = new ObjectDoubleOpenHashMap<>();

    List<Object> presentTargets = classSet.getTargets();
    DoubleArray ltRelativeFrequency = Arrays.newDoubleArray(presentTargets.size());
    DoubleArray gtRelativeFrequency = Arrays.newDoubleArray(presentTargets.size());



    double ltWeight = 0.0, gtWeight = 0.0;

    // Initialize all value to the right (i.e. all getPosteriorProbabilities are larger than the
    // initial threshold)
    for (ClassSet.Sample sample : classSet.samples()) {
      double weight = sample.getWeight();
      gtWeight += weight;

      lt.put(sample.getTarget(), 0);
      gt.put(sample.getTarget(), weight);
    }


    // Transfer weights from the initial example
    Example first = distances.get(0).example;
    Object prevTarget = y.loc().get(Object.class, first.getIndex());
    gt.addTo(prevTarget, -first.getWeight());
    lt.addTo(prevTarget, first.getWeight());
    gtWeight -= first.getWeight();
    ltWeight += first.getWeight();

    double prevDistance = distances.get(0).distance;
    double lowestImpurity = Double.POSITIVE_INFINITY;
    double threshold = distances.get(0).distance / 2;
    Gain gain = getGain();
    double ltGap = 0.0, gtGap = distanceSum, largestGap = Double.NEGATIVE_INFINITY;
    for (int i = 1; i < distances.size(); i++) {
      ExampleDistance ed = distances.get(i);
      Object target = y.loc().get(Object.class, ed.example.getIndex());


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
          threshold = (ed.distance + prevDistance) / 2;
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

      ltGap += ed.distance;
      gtGap -= ed.distance;

      prevDistance = ed.distance;
      prevTarget = target;
    }

    double minimumMargin = Double.POSITIVE_INFINITY;
    return new Threshold(threshold, lowestImpurity, largestGap, minimumMargin);
  }

  /**
   * Basic implementation of the splitting procedure
   *
   * @param distanceMap the distance map (mapping examples to distances to the shapelet)
   * @param classSet the examples
   * @param threshold the threshold
   * @param shapelet the shapelet
   * @return the examples . split
   */
  protected TreeSplit<ShapeletThreshold> split(IntDoubleMap distanceMap, ClassSet classSet,
      double threshold, Shapelet shapelet) {
    ClassSet left = new ClassSet(classSet.getDomain());
    ClassSet right = new ClassSet(classSet.getDomain());

    /*
     * Partition every class separately
     */
    for (ClassSet.Sample sample : classSet.samples()) {
      Object target = sample.getTarget();

      ClassSet.Sample leftSample = ClassSet.Sample.create(target);
      ClassSet.Sample rightSample = ClassSet.Sample.create(target);

      /*
       * STEP 1: Partition the examples according to threshold
       */
      for (Example example : sample) {
        double shapeletDistance = distanceMap.get(example.getIndex());
        if (shapeletDistance < threshold) {
          leftSample.add(example);
        } else {
          rightSample.add(example);
        }
      }

      /*
       * STEP 3: Ignore classes with no examples in the partition
       */
      if (!leftSample.isEmpty()) {
        left.add(leftSample);
      }
      if (!rightSample.isEmpty()) {
        right.add(rightSample);
      }
    }

    return new TreeSplit<>(left, right, new ShapeletThreshold(shapelet, threshold));
  }



  /**
   * The type Example distance.
   */
  private static class ExampleDistance implements Comparable<ExampleDistance> {
    /**
     * The Distance.
     */
    public final double distance;
    /**
     * The Example.
     */
    public final Example example;

    public ExampleDistance(double distance, Example example) {
      this.distance = distance;
      this.example = example;
    }

    @Override
    public int compareTo(ExampleDistance o) {
      return Double.compare(distance, o.distance);
    }

    @Override
    public String toString() {
      return String.format("ExampleDistance(id=%d, %.2f)", example.getIndex(), distance);
    }
  }

  public static class Builder {

    private int shapelets = 10;
    private int sampleSize = -1;
    private Gain gain = Gain.INFO;
    private int upper = -1;
    private int lower = 2;
    private Distance distance;
    private double alpha;

    private Builder(Distance distance) {
      this.distance = distance;
    }

    public Builder withDistance(Distance distance) {
      this.distance = distance;
      return this;
    }

    public Builder withUpperLength(int upper) {
      this.upper = upper;
      return this;
    }

    public Builder withLowerLength(int lower) {
      this.lower = lower;
      return this;
    }

    public Builder withSampleSize(int sampleSize) {
      this.sampleSize = sampleSize;
      return this;
    }

    public Builder withAlpha(double alpha) {
      this.alpha = alpha;
      return this;
    }

    public Builder withInspectedShapelets(int maxShapelets) {
      this.shapelets = maxShapelets;
      return this;
    }

    public RandomShapeletSplitter build() {
      return new AggregateRandomShapeletSplitter(this);
    }
  }

  protected static class Threshold {
    public final double threshold, impurity, gap, margin;

    public Threshold(double threshold, double impurity, double gap, double margin) {
      this.threshold = threshold;
      this.impurity = impurity;
      this.gap = gap;
      this.margin = margin;
    }

    public static Threshold inf() {
      return new Threshold(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
          Double.NEGATIVE_INFINITY);
    }

    public boolean isBetterThan(Threshold bestThreshold) {
      return this.impurity < bestThreshold.impurity
          || (this.impurity == bestThreshold.impurity && this.gap > bestThreshold.gap);
    }
  }
}
