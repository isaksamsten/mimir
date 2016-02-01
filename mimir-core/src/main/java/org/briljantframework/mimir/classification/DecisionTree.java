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

import java.util.Collections;
import java.util.Set;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Convert;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Splitter;
import org.briljantframework.mimir.classification.tree.TreeBranch;
import org.briljantframework.mimir.classification.tree.TreeClassifier;
import org.briljantframework.mimir.classification.tree.TreeLeaf;
import org.briljantframework.mimir.classification.tree.TreeNode;
import org.briljantframework.mimir.classification.tree.TreeSplit;
import org.briljantframework.mimir.classification.tree.TreeVisitor;
import org.briljantframework.mimir.classification.tree.ValueThreshold;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DecisionTree extends TreeClassifier<ValueThreshold> {

  private final int depth;

  private DecisionTree(Vector classes, TreeNode<ValueThreshold> node, int depth,
      TreeVisitor<ValueThreshold> predictionVisitor) {
    super(classes, node, predictionVisitor);
    this.depth = depth;
  }

  public int getDepth() {
    return depth;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner implements Predictor.Learner<DecisionTree> {

    protected final double mininumWeight = 1;
    protected final Splitter splitter;

    protected ClassSet classSet;
    protected Vector classes = null;

    public Learner(Splitter splitter) {
      this(splitter, null, null);
    }


    protected Learner(Splitter splitter, ClassSet classSet, Vector classes) {
      this.splitter = splitter;
      this.classSet = classSet;
      this.classes = classes;
    }

    @Override
    public DecisionTree fit(DataFrame x, Vector y) {
      ClassSet classSet = this.classSet;
      Vector classes = this.classes != null ? this.classes : Vectors.unique(y);
      if (classSet == null) {
        classSet = new ClassSet(y, classes);
      }

      Params p = new Params();
      TreeNode<ValueThreshold> node = build(x, y, p, classSet);
      return new DecisionTree(classes, node, p.depth, new SimplePredictionVisitor());
    }

    protected TreeNode<ValueThreshold> build(DataFrame frame, Vector target, Params p,
        ClassSet classSet) {
      return build(frame, target, p, classSet, 1);
    }

    protected TreeNode<ValueThreshold> build(DataFrame frame, Vector target, Params p,
        ClassSet classSet, int depth) {

      if (classSet.getTotalWeight() <= mininumWeight || classSet.getTargetCount() == 1) {
        p.depth = Math.max(p.depth, depth);
        return TreeLeaf.fromExamples(classSet);
      }
      TreeSplit<ValueThreshold> maxSplit = splitter.find(classSet, frame, target);
      if (maxSplit == null) {
        p.depth = Math.max(p.depth, depth);
        return TreeLeaf.fromExamples(classSet);
      } else {
        ClassSet left = maxSplit.getLeft();
        ClassSet right = maxSplit.getRight();
        if (left.isEmpty()) {
          p.depth = Math.max(p.depth, depth);
          return TreeLeaf.fromExamples(right);
        } else if (right.isEmpty()) {
          p.depth = Math.max(p.depth, depth);
          return TreeLeaf.fromExamples(left);
        } else {
          TreeNode<ValueThreshold> leftNode = build(frame, target, p, left, depth + 1);
          TreeNode<ValueThreshold> rightNode = build(frame, target, p, right, depth + 1);
          return new TreeBranch<>(leftNode, rightNode, classes, maxSplit.getThreshold(), 1);
        }
      }
    }

    private final class Params {

      public int depth = 0;
    }
  }

  private static final class SimplePredictionVisitor implements TreeVisitor<ValueThreshold> {

    private static final int MISSING = 0, LEFT = -1, RIGHT = 1;

    @Override
    public DoubleArray visitLeaf(TreeLeaf<ValueThreshold> leaf, Vector example) {
      return leaf.getProbabilities();
    }

    @Override
    public DoubleArray visitBranch(TreeBranch<ValueThreshold> node, Vector example) {
      Object threshold = node.getThreshold().getValue();
      int axis = node.getThreshold().getAxis();
      int direction = MISSING;
      if (!example.loc().isNA(axis)) {
        if (Is.nominal(threshold)) {
          direction = example.loc().get(Object.class, axis).equals(threshold) ? LEFT : RIGHT;
        } else {
          // note: Is.nominal return true for any non-number and Number is always comparable
          // @SuppressWarnings("unchecked")
          // Comparable<Object> leftComparable = example.loc().get(Comparable.class, axis);
          // direction = leftComparable.compareTo(threshold) <= 0 ? LEFT : RIGHT;
          double left = example.loc().getAsDouble(axis);
          double right = Convert.to(Double.class, threshold);
          direction = Double.compare(left, right) <= 0 ? LEFT : RIGHT;
        }
      }

      switch (direction) {
        case LEFT:
          return visit(node.getLeft(), example);
        case RIGHT:
          return visit(node.getRight(), example);
        case MISSING:
        default:
          return visit(node.getLeft(), example); // TODO: what to do with missing values?
      }
    }
  }
}
