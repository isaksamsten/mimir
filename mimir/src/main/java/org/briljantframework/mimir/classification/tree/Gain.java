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

package org.briljantframework.mimir.classification.tree;

import org.briljantframework.array.DoubleArray;

/**
 * Created by Isak Karlsson on 08/09/14.
 */
public class Gain {

  public static final Gain GINI = Gain.with(Gini.getInstance());
  public static final Gain INFO = Gain.with(Entropy.getInstance());

  private final Impurity impurity;

  private Gain(Impurity impurity) {
    this.impurity = impurity;
  }

  public static Gain with(Impurity impurity) {
    return new Gain(impurity);
  }

  public Impurity getImpurity() {
    return impurity;
  }

  public double compute(TreeSplit<?> split) {
    ClassSet left = split.getLeft();
    ClassSet right = split.getRight();
    return compute(left.getTotalWeight(), left.getRelativeFrequencies(), right.getTotalWeight(),
        right.getRelativeFrequencies());
  }

  public double compute(double leftWeight, DoubleArray left, double rightWeight, DoubleArray right) {
    double totalWeight = leftWeight + rightWeight;
    if (leftWeight > 0) {
      leftWeight = (leftWeight / totalWeight) * impurity.impurity(left);// * leftWeight;
    } else {
      leftWeight = 0.0;
    }

    if (rightWeight > 0) {
      rightWeight = (rightWeight / totalWeight) * impurity.impurity(right);// * rightWeight;
    } else {
      rightWeight = 0.0;
    }

    return leftWeight + rightWeight;
  }
}
