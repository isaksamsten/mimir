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
import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;

/**
 * NOTE: This cannot be reused among trees (it is stateful for performance reasons)
 * <p>
 * Created by Isak Karlsson on 09/09/14.
 */
public abstract class RandomSplitter extends AbstractSplitter<Instance> {

  public static Splitter<Instance> log() {
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        return (int) Math.round(Math.log(featureSize) / Math.log(2)) + 1;
      }
    };
  }

  public static Splitter<Instance> sqrt() {
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        return (int) Math.round(Math.sqrt(featureSize)) + 1;
      }
    };
  }

  public static Splitter<Instance> all() {
    return new RandomSplitter(Gain.INFO) {
      @Override
      protected int getFeatureInspection(int featureSize) {
        return featureSize;
      }
    };
  }

  public static Splitter<Instance> newInstance(int size) {
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
  public TreeSplit<Instance> find(ClassSet classSet, Input<? extends Instance> x, List<?> y) {

    int featureSize = ((MultidimensionalSchema) x.getSchema()).attributes();
    int noInspected = getFeatureInspection(featureSize);
    TreeSplit<Instance> bestSplit = null;
    double bestImpurity = Double.POSITIVE_INFINITY;
    PermuteIndexIterable iterator = new PermuteIndexIterable(featureSize, noInspected);
    for (Integer axis : iterator) {
      TreeNodeTest<Instance> thresholds = findThresholds(x, axis, classSet);
      TreeSplit<Instance> split = split(x, classSet, thresholds);
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

  protected TreeNodeTest<Instance> findThresholds(Input<? extends Instance> input, int axis,
      ClassSet classSet) {
    MultidimensionalSchema schema = (MultidimensionalSchema) input.getSchema();
    if (schema.isNumericalAttribute(axis)) {
      return sampleNumericValue(schema, input, axis, classSet);
    } else {
      return sampleCategoricValue(schema, input, axis, classSet);
    }
  }

  protected TreeNodeTest<Instance> sampleNumericValue(MultidimensionalSchema schema,
      Input<? extends Instance> in, int axis, ClassSet classSet) {
    Example a = classSet.getRandomSample().getRandomExample();
    Example b = classSet.getRandomSample().getRandomExample();
    Instance exa = in.get(a.getIndex());
    Instance exb = in.get(b.getIndex());

    double valueA = exa.getNumericalAttribute(axis);
    double valueB = exb.getNumericalAttribute(axis);

    // TODO - what if both A and B are missing?
    double value;
    if (Is.NA(valueA)) {
      value = valueB;
    } else if (Is.NA(valueB)) {
      value = valueA;
    } else {
      value = (valueA + valueB) / 2;
    }
    return new NumericalNodeTest(axis, value);
  }

  /**
   * Sample categoric value.
   *
   * @param schema
   * @param classSet the examples
   * @return the value
   */
  protected TreeNodeTest<Instance> sampleCategoricValue(MultidimensionalSchema schema,
      Input<? extends Instance> in, int axis, ClassSet classSet) {
    int ax = axis - schema.numericalAttributes();
    Example example = classSet.getRandomSample().getRandomExample();
    Object categoricalAttribute = in.get(example.getIndex()).getCategoricalAttribute(ax);
    return new CategoricalNodeTest(ax, categoricalAttribute);
  }

}
