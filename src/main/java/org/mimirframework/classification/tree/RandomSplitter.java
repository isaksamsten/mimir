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

package org.mimirframework.classification.tree;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.primitive.ArrayAllocations;

/**
 * NOTE: This cannot be reused among trees (it is stateful for performance reasons)
 * <p>
 * Created by Isak Karlsson on 09/09/14.
 */
public class RandomSplitter extends AbstractSplitter {

  private final int maxFeatures;

  private final org.mimirframework.classification.tree.Gain criterion;
  private int[] features = null;

  public RandomSplitter(int maxFeatures, org.mimirframework.classification.tree.Gain criterion) {
    this.maxFeatures = maxFeatures;
    this.criterion = criterion;
  }

  public RandomSplitter(int maxFeatures) {
    this(maxFeatures, org.mimirframework.classification.tree.Gain.INFO);
  }

  public static Builder withMaximumFeatures(int maxFeatures) {
    return new Builder(maxFeatures);
  }

  @Override
  public org.mimirframework.classification.tree.TreeSplit<ValueThreshold> find(org.mimirframework.classification.tree.ClassSet classSet, DataFrame dataFrame, Vector column) {
    if (features == null) {
      initialize(dataFrame);
    }

    int maxFeatures =
        this.maxFeatures > 0 ? this.maxFeatures
            : (int) Math.round(Math.sqrt(dataFrame.columns())) + 1;

    // TODO! Fix me!
    synchronized (features) {
      ArrayAllocations.shuffle(features);
    }

    org.mimirframework.classification.tree.TreeSplit<ValueThreshold> bestSplit = null;
    double bestImpurity = Double.POSITIVE_INFINITY;
    for (int i = 0; i < features.length && i < maxFeatures; i++) {
      int axis = features[i];

      Object threshold = search(dataFrame.loc().get(axis), classSet);
      if (Is.NA(threshold)) {
        continue;
      }

      org.mimirframework.classification.tree.TreeSplit<ValueThreshold> split = split(dataFrame, classSet, axis, threshold);
      double impurity = criterion.compute(split);
      if (impurity < bestImpurity) {
        bestSplit = split;
        bestImpurity = impurity;
      }
    }

    if (bestSplit != null) {
      bestSplit.setImpurity(bestImpurity);
    }
    return bestSplit;
  }

  private void initialize(DataFrame dataFrame) {
    this.features = new int[dataFrame.columns()];
    for (int i = 0; i < features.length; i++) {
      this.features[i] = i;
    }
  }

  /**
   * Search value.
   *
   * @param axis the dataset
   * @param classSet the examples
   * @return the value
   */
  protected Object search(Vector axis, org.mimirframework.classification.tree.ClassSet classSet) {
    switch (axis.getType().getScale()) {
      case NOMINAL:
        return sampleCategoricValue(axis, classSet);
      case NUMERICAL:
        return sampleNumericValue(axis, classSet);
      default:
        throw new IllegalStateException(String.format("Header: %s, not supported", axis.getType()));
    }
  }

  /**
   * Sample numeric value.
   *
   * @param vector the dataset
   * @param classSet the examples
   * @return the value
   */
  protected double sampleNumericValue(Vector vector, org.mimirframework.classification.tree.ClassSet classSet) {
    org.mimirframework.classification.tree.Example a = classSet.getRandomSample().getRandomExample();
    org.mimirframework.classification.tree.Example b = classSet.getRandomSample().getRandomExample();

    double valueA = vector.loc().getAsDouble(a.getIndex());
    double valueB = vector.loc().getAsDouble(b.getIndex());

    // TODO - what if both A and B are missing?
    if (Is.NA(valueA)) {
      return valueB;
    } else if (Is.NA(valueB)) {
      return valueB;
    } else {
      return (valueA + valueB) / 2;
    }

  }

  /**
   * Sample categoric value.
   *
   * @param axisVector the dataset
   * @param classSet the examples
   * @return the value
   */
  protected Object sampleCategoricValue(Vector axisVector, org.mimirframework.classification.tree.ClassSet classSet) {
    org.mimirframework.classification.tree.Example example = classSet.getRandomSample().getRandomExample();
    return axisVector.get(Object.class, example.getIndex());
  }

  /**
   * The type Builder.
   */
  public static class Builder {

    private int maxFeatures;
    private org.mimirframework.classification.tree.Gain criterion = org.mimirframework.classification.tree.Gain.INFO;

    private Builder(int maxFeatures) {
      this.maxFeatures = maxFeatures;
    }

    /**
     * Sets max features.
     *
     * @param maxFeatures the max features
     * @return the max features
     */
    public Builder setMaximumFeatures(int maxFeatures) {
      this.maxFeatures = maxFeatures;
      return this;
    }

    /**
     * Sets withCriterion.
     *
     * @param criterion the withCriterion
     * @return the withCriterion
     */
    public Builder withCriterion(org.mimirframework.classification.tree.Gain criterion) {
      this.criterion = criterion;
      return this;
    }

    /**
     * Create random splitter.
     *
     * @return the random splitter
     */
    public RandomSplitter create() {
      return new RandomSplitter(maxFeatures, criterion);
    }

  }
}
