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
package org.briljantframework.mimir.classification.tree;

import java.util.List;

import org.briljantframework.data.Is;
import org.briljantframework.mimir.data.Dataset;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Instance;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.util.primitive.ArrayAllocations;

/**
 * NOTE: This cannot be reused among trees (it is stateful for performance reasons)
 * <p>
 * Created by Isak Karlsson on 09/09/14.
 */
public class RandomSplitter extends AbstractSplitter {

  private final int maxFeatures;

  private final Gain criterion;
  private int[] features = null;

  public RandomSplitter(int maxFeatures) {
    this(maxFeatures, Gain.INFO);
  }

  public RandomSplitter(int maxFeatures, Gain criterion) {
    this.maxFeatures = maxFeatures;
    this.criterion = criterion;
  }

  public static Builder withMaximumFeatures(int maxFeatures) {
    return new Builder(maxFeatures);
  }

  @Override
  public TreeSplit<ValueThreshold> find(ClassSet classSet, Input<? extends Instance> x,
      Output<?> y) {
    int[] features = initialize(x);
    int maxFeatures =
        this.maxFeatures > 0 ? this.maxFeatures : (int) Math.round(Math.sqrt(x.get(0).size())) + 1;
    TreeSplit<ValueThreshold> bestSplit = null;
    double bestImpurity = Double.POSITIVE_INFINITY;
    for (int i = 0; i < features.length && i < maxFeatures; i++) {
      int axis = features[i];

      Object threshold = search(x, axis, classSet);
      if (Is.NA(threshold)) {
        continue;
      }

      TreeSplit<ValueThreshold> split = split(x, classSet, axis, threshold);
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

  private int[] initialize(Input<? extends Instance> dataFrame) {
    int[] features = new int[dataFrame.get(0).size()];
    for (int i = 0; i < features.length; i++) {
      features[i] = i;
    }
    ArrayAllocations.shuffle(features);
    return features;
  }

  /**
   * Search value.
   *
   * @param axis the dataset
   * @param classSet the examples
   * @return the value
   */
  protected Object search(Input<? extends Instance> input, int axis, ClassSet classSet) {
    List<Class<?>> types = input.getProperties().get(Dataset.FEATURE_TYPES);
    boolean i = isNumeric(types.get(axis));
    if (i) {
      return sampleNumericValue(input, axis, classSet);
    } else {
      return sampleCategoricValue(input, axis, classSet);
    }
  }


  private boolean isNumeric(Class<?> cls) {
    return cls != null && Number.class.isAssignableFrom(cls);
  }

  /**
   * Sample numeric value.
   *
   * @param classSet the examples
   * @return the value
   */
  protected double sampleNumericValue(Input<? extends Instance> in, int axis, ClassSet classSet) {
    Example a = classSet.getRandomSample().getRandomExample();
    Example b = classSet.getRandomSample().getRandomExample();
    Instance exa = in.get(a.getIndex());
    Instance exb = in.get(b.getIndex());

    double valueA = exa.getDouble(axis);
    double valueB = exb.getDouble(axis);

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
   * @param classSet the examples
   * @return the value
   */
  protected Object sampleCategoricValue(Input<? extends Instance> in, int axis, ClassSet classSet) {
    Example example = classSet.getRandomSample().getRandomExample();
    return in.get(example.getIndex()).get(axis);
  }

  /**
   * The type Builder.
   */
  public static class Builder {

    private int maxFeatures;
    private Gain criterion = Gain.INFO;

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
    public Builder withCriterion(Gain criterion) {
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
