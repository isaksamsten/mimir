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

import java.util.List;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.tree.*;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.supervised.AbstractLearner;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * Created by isak on 11/16/15.
 */
public class HyperPlaneTree extends TreeClassifier<Instance> {
  protected HyperPlaneTree(List<?> classes, TreeVisitor<Instance, ?> predictionVisitor) {
    super(classes, predictionVisitor);
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

  private static class HyperPlaneTreeVisitor implements TreeVisitor<Instance, HyperPlaneThreshold> {
    private final TreeNode<Instance, HyperPlaneThreshold> root;

    public HyperPlaneTreeVisitor(TreeNode<Instance, HyperPlaneThreshold> root) {
      this.root = root;
    }

    @Override
    public TreeNode<Instance, HyperPlaneThreshold> getRoot() {
      return root;
    }

    @Override
    public DoubleArray visitLeaf(TreeLeaf<Instance, HyperPlaneThreshold> leaf, Instance example) {
      return leaf.getProbabilities();
    }

    @Override
    public DoubleArray visitBranch(TreeBranch<Instance, HyperPlaneThreshold> node,
        Instance example) {
      DoubleArray row = DoubleArray.zeros(example.size() + 1);
      row.set(0, 1);
      for (int i = 1; i < example.size(); i++) {
        row.set(i, example.getAsDouble(i));
      }
      row = take(row, node.getThreshold().getFeatures());
      DoubleArray weights = node.getThreshold().getWeights();
      if (Arrays.inner(row, weights) < node.getThreshold().getThreshold()) {
        return visit(node.getLeft(), example);
      } else {
        return visit(node.getRight(), example);
      }
    }
  }

  public static class Learner extends AbstractLearner<Instance, Object, HyperPlaneTree> {
    private final ClassSet set;
    private final Gain criterion = Gain.INFO;
    private final List<?> classes;
    private final int noHyperPlanes;

    public Learner(ClassSet set, List<?> classes, int noHyperPlanes) {
      this.set = set;
      this.classes = classes;
      this.noHyperPlanes = noHyperPlanes;
    }

    @Override
    public HyperPlaneTree fit(Input<? extends Instance> x, Output<?> y) {
      ClassSet set = this.set;
      List<?> classes = this.classes == null ? Outputs.unique(y) : this.classes;
      if (set == null) {
        set = new ClassSet(y, classes);
      }

      DoubleArray array = Arrays.hstack(DoubleArray.ones(x.size(), 1), Inputs.toDoubleArray(x));
      TreeNode<Instance, HyperPlaneThreshold> root = build(array, y, set);
      return new HyperPlaneTree(classes, new HyperPlaneTreeVisitor(root));
    }

    private TreeNode<Instance, HyperPlaneThreshold> build(DoubleArray x, Output<?> y,
        ClassSet set) {
      if (set.getTotalWeight() <= 1.0 || set.getTargetCount() == 1) {
        return TreeLeaf.fromExamples(set);
      }
      TreeSplit<HyperPlaneThreshold> maxSplit = findHyperPlaneRandomMeanPoint(set, x);
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
          TreeNode<Instance, HyperPlaneThreshold> leftNode = build(x, y, left);
          TreeNode<Instance, HyperPlaneThreshold> rightNode = build(x, y, right);
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
        DoubleArray x) {
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
