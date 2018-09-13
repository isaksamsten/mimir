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

import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.series.Series;

/**
 * @author Isak Karlsson
 */
public final class TreeLeaf<In> implements TreeNode<In> {

  private final Array<?> domain;
  private final DoubleArray probabilities;
  private final double weight;

  public TreeLeaf(Array<?> domain, DoubleArray probabilities, double weight) {
    this.domain = domain;
    this.probabilities = probabilities;
    this.weight = weight;
  }

  public static <In> TreeLeaf<In> fromExamples(ClassSet classSet) {
    return fromExamples(classSet, 1);
  }

  public static <In> TreeLeaf<In> fromExamples(ClassSet classSet, double weight) {
    Array<?> domain = classSet.getDomain();
    DoubleArray prob = DoubleArray.zeros(domain.size());
    double totalWeight = classSet.getTotalWeight();
    for (int i = 0; i < domain.size(); i++) {
      Object label = domain.get(i);
      ClassSet.Sample sample = classSet.get(label);
      if (sample == null) {
        prob.set(i, 0);
      } else {
        prob.set(i, sample.getWeight() / totalWeight);
      }
    }
    return new TreeLeaf<>(domain, prob, weight);
  }

  public DoubleArray getProbabilities() {
    return probabilities;
  }

  @Override
  public double getWeight() {
    return weight;
  }

  public Array<?> getDomain() {
    return domain;
  }

  @Override
  public Series getClassDistribution() {
    Series.Builder dist = Series.Builder.of(double.class);
    for (int i = 0; i < probabilities.size(); i++) {
      dist.set(getDomain().get(i), probabilities.get(i));
    }
    return dist.build();
  }

  @Override
  public final DoubleArray visit(TreeVisitor<In> visitor, In example) {
    return visitor.visitLeaf(this, example);
  }


}
