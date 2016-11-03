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

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.data.Dataset;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Instance;
import org.briljantframework.mimir.data.Output;

/**
 * NOTE: This cannot be reused among trees (it is stateful for performance reasons)
 * <p>
 * Created by Isak Karlsson on 09/09/14.
 */
public abstract class RandomSplitter extends AbstractSplitter {

  public static Splitter log() {
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        return (int) Math.round(Math.log(featureSize) / Math.log(2)) + 1;
      }
    };
  }

  public static Splitter sqrt() {
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        return (int) Math.round(Math.sqrt(featureSize)) + 1;
      }
    };
  }

  public static Splitter all() {
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        return featureSize;
      }
    };
  }

  public static Splitter newInstance(int size) {
    Check.argument(size > 0, "illegal feature size");
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        if (size > featureSize) {
          throw new IllegalStateException("illegal feature size");
        }
        return size;
      }
    };
  }

  private final Gain criterion;

  private RandomSplitter(Gain criterion) {
    this.criterion = Objects.requireNonNull(criterion);
  }

  protected abstract int getFeatureInspection(int featureSize);

  @Override
  public TreeSplit<ValueThreshold> find(ClassSet classSet, Input<? extends Instance> x,
      Output<?> y) {

    int featureSize = x.getProperty(Dataset.FEATURE_SIZE);
    int noInspected = getFeatureInspection(featureSize);
    TreeSplit<ValueThreshold> bestSplit = null;
    double bestImpurity = Double.POSITIVE_INFINITY;
    PermuteIndexIterable iterator = new PermuteIndexIterable(featureSize, noInspected);
    for (Integer axis : iterator) {
      Collection<Object> thresholds = findThresholds(x, axis, classSet);
      for (Object threshold : thresholds) {
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
    }

    if (bestSplit != null) {
      bestSplit.setImpurity(bestImpurity);
    }
    return bestSplit;
  }

  protected Collection<Object> findThresholds(Input<? extends Instance> input, int axis,
      ClassSet classSet) {
    List<Class<?>> types = input.getProperty(Dataset.FEATURE_TYPES);
    if (Is.numeric(types.get(axis))) {
      return sampleNumericValue(input, axis, classSet);
    } else {
      return sampleCategoricValue(input, axis, classSet);
    }
  }

  protected Collection<Object> sampleNumericValue(Input<? extends Instance> in, int axis,
      ClassSet classSet) {
    Example a = classSet.getRandomSample().getRandomExample();
    Example b = classSet.getRandomSample().getRandomExample();
    Instance exa = in.get(a.getIndex());
    Instance exb = in.get(b.getIndex());

    double valueA = exa.getDouble(axis);
    double valueB = exb.getDouble(axis);

    // TODO - what if both A and B are missing?
    if (Is.NA(valueA)) {
      return Collections.singleton(valueB);
    } else if (Is.NA(valueB)) {
      return Collections.singleton(valueB);
    } else {
      return Collections.singleton((valueA + valueB) / 2);
    }
  }

  /**
   * Sample categoric value.
   *
   * @param classSet the examples
   * @return the value
   */
  protected Collection<Object> sampleCategoricValue(Input<? extends Instance> in, int axis,
      ClassSet classSet) {
    Example example = classSet.getRandomSample().getRandomExample();
    return Collections.singleton(in.get(example.getIndex()).get(axis));
  }
}
