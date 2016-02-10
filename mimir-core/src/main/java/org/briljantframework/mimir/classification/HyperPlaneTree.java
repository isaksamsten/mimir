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

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.classification.tree.Gain;
import org.briljantframework.mimir.classification.tree.HyperPlaneThreshold;
import org.briljantframework.mimir.classification.tree.TreeBranch;
import org.briljantframework.mimir.classification.tree.TreeClassifier;
import org.briljantframework.mimir.classification.tree.TreeLeaf;
import org.briljantframework.mimir.classification.tree.TreeNode;
import org.briljantframework.mimir.classification.tree.TreeSplit;
import org.briljantframework.mimir.classification.tree.TreeVisitor;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * Created by isak on 11/16/15.
 */
public class HyperPlaneTree extends TreeClassifier<HyperPlaneThreshold> {
  protected HyperPlaneTree(Vector classes, TreeNode<HyperPlaneThreshold> node,
      TreeVisitor<HyperPlaneThreshold> predictionVisitor) {
    super(classes, node, predictionVisitor);
  }

  private static DoubleArray take(DoubleArray x, IntArray i) {
    if (i == null) {
      return x;
    }
    DoubleArray y = DoubleArray.zeros(i.size() + 1);
    y.set(0, x.get(0));
    for (int j = 1; j < i.size(); j++) {
      y.set(j, x.get(i.get(j)));
    }
    return y;
  }

  private static IntArray getRandomFeatures(DoubleArray x) {
    IntArray features = Arrays.range(1, x.size()).copy();
    features.permute(x.size() - 1);
    // double thing = Math.log(x.size()) / Math.log(2);
    double thing = Math.sqrt(x.size());
    int end = (int) (Math.round(thing) + 1);
    return features.get(Arrays.range(0, end));
  }

  private static class HyperPlaneTreeVisitor implements TreeVisitor<HyperPlaneThreshold> {
    @Override
    public DoubleArray visitLeaf(TreeLeaf<HyperPlaneThreshold> leaf, Vector example) {
      return leaf.getProbabilities();
    }

    @Override
    public DoubleArray visitBranch(TreeBranch<HyperPlaneThreshold> node, Vector example) {
      DoubleArray row = DoubleArray.zeros(example.size() + 1);
      row.set(0, 1);
      Vectors.copy(example, row.getView(Range.of(1, row.size())));

      row = take(row, node.getThreshold().getFeatures());
      DoubleArray weights = node.getThreshold().getWeights();
      if (Arrays.inner(row, weights) < node.getThreshold().getThreshold()) {
        return visit(node.getLeft(), example);
      } else {
        return visit(node.getRight(), example);
      }
    }
  }

  public static class Learner implements Predictor.Learner<HyperPlaneTree> {
    private final ClassSet set;
    private final Gain criterion = Gain.INFO;
    private final Vector classes;
    private final int noHyperPlanes;

    public Learner(ClassSet set, Vector classes, int noHyperPlanes) {
      this.set = set;
      this.classes = classes;
      this.noHyperPlanes = noHyperPlanes;
    }

    @Override
    public HyperPlaneTree fit(DataFrame x, Vector y) {
      ClassSet set = this.set;
      Vector classes = this.classes == null ? Vectors.unique(y) : this.classes;
      if (set == null) {
        set = new ClassSet(y, classes);
      }

      DoubleArray array = Arrays.hstack(DoubleArray.ones(x.rows(), 1), DataFrames.toDoubleArray(x));
      TreeNode<HyperPlaneThreshold> root = build(array, y, set);
      return new HyperPlaneTree(classes, root, new HyperPlaneTreeVisitor());
    }

    private TreeNode<HyperPlaneThreshold> build(DoubleArray x, Vector y, ClassSet set) {
      if (set.getTotalWeight() <= 1.0 || set.getTargetCount() == 1) {
        return TreeLeaf.fromExamples(set);
      }
      TreeSplit<HyperPlaneThreshold> maxSplit = findHyperPlaneRandomMeanPoint(set, x, y);
      if (maxSplit == null) {
        return TreeLeaf.fromExamples(set);
      } else {
        ClassSet left = maxSplit.getLeft();
        ClassSet right = maxSplit.getRight();
        if (left.isEmpty()) {
          return TreeLeaf.fromExamples(right);
        } else if (right.isEmpty()) {
          return TreeLeaf.fromExamples(left);
        } else {
          TreeNode<HyperPlaneThreshold> leftNode = build(x, y, left);
          TreeNode<HyperPlaneThreshold> rightNode = build(x, y, right);
          return new TreeBranch<>(leftNode, rightNode, classes, maxSplit.getThreshold(), 1);
        }
      }
    }

    private TreeSplit<HyperPlaneThreshold> findHyperPlaneRandomPoint(ClassSet set, DoubleArray x,
        Vector y) {
      TreeSplit<HyperPlaneThreshold> bestSplit = null;
      double bestImpurity = Double.POSITIVE_INFINITY;
      for (int i = 0; i < noHyperPlanes; i++) {
        int index = set.getRandomSample().getRandomExample().getIndex();
        DoubleArray row = x.getRow(index);
        IntArray randomFeatures = getRandomFeatures(row);
        DoubleArray weights = Arrays.randn(randomFeatures.size() + 1);
        double threshold = Arrays.inner(take(row, randomFeatures), weights);
        TreeSplit<HyperPlaneThreshold> split = split(x, set, weights, randomFeatures, threshold);
        double impurity = criterion.compute(split);
        if (impurity < bestImpurity) {
          bestImpurity = impurity;
          bestSplit = split;
        }
      }
      if (bestSplit != null) {
        bestSplit.setImpurity(bestImpurity);
      }

      return bestSplit;
    }

    protected TreeSplit<HyperPlaneThreshold> split(DoubleArray x, ClassSet set, DoubleArray weights,
        IntArray dims, double threshold) {
      ClassSet left = new ClassSet(set.getDomain());
      ClassSet right = new ClassSet(set.getDomain());

      /*
       * Partition every class separately
       */
      for (ClassSet.Sample sample : set.samples()) {
        Object target = sample.getTarget();

        ClassSet.Sample leftSample = ClassSet.Sample.create(target);
        ClassSet.Sample rightSample = ClassSet.Sample.create(target);
        for (Example example : sample) {
          DoubleArray row = x.getRow(example.getIndex());
          row = take(row, dims);
          if (Arrays.inner(row, weights) < threshold) {
            leftSample.add(example);
          } else {
            rightSample.add(example);
          }
        }

        if (!leftSample.isEmpty()) {
          left.add(leftSample);
        }
        if (!rightSample.isEmpty()) {
          right.add(rightSample);
        }
      }

      return new TreeSplit<>(left, right, new HyperPlaneThreshold(weights, dims, threshold));
    }

    private TreeSplit<HyperPlaneThreshold> findHyperPlaneRandomMeanPoint(ClassSet set,
        DoubleArray x, Vector y) {
      TreeSplit<HyperPlaneThreshold> bestSplit = null;
      double bestImpurity = Double.POSITIVE_INFINITY;
      for (int i = 0; i < noHyperPlanes; i++) {
        int a = set.getRandomSample().getRandomExample().getIndex();
        int b = set.getRandomSample().getRandomExample().getIndex();
        DoubleArray ra = x.getRow(a);
        DoubleArray rb = x.getRow(b);

        IntArray randomFeatures = getRandomFeatures(ra);
        DoubleArray weights = Arrays.randn(randomFeatures.size() + 1);
        double threshold = Arrays.inner(take(ra.plus(rb).div(2), randomFeatures), weights);
        TreeSplit<HyperPlaneThreshold> split = split(x, set, weights, randomFeatures, threshold);
        double impurity = criterion.compute(split);
        if (impurity < bestImpurity) {
          bestImpurity = impurity;
          bestSplit = split;
        }
      }
      if (bestSplit != null) {
        bestSplit.setImpurity(bestImpurity);
      }

      return bestSplit;
    }
  }
}
