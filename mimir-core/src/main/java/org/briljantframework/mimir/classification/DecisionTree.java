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

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.series.Convert;
import org.briljantframework.mimir.classification.tree.*;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.supervised.AbstractLearner;
import org.briljantframework.mimir.supervised.Characteristic;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DecisionTree<Out> extends TreeClassifier<Instance, Out> {

  public static Property<Double> MIN_LEAF_SIZE = Property.of("min_leaf_size", Double.class, 1.0);
  public static Property<Splitter> SPLITTER = Property.of("splitter", Splitter.class);
  public static Property<Integer> MAX_DEPTH =
      Property.of("max_depth", Integer.class, Integer.MAX_VALUE, i -> i > 0);

  private final int depth;

  private DecisionTree(Array<Out> classes, int depth,
      TreeVisitor<Instance, ValueThreshold> predictionVisitor) {
    super(classes, predictionVisitor);
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
  public static class Learner<Out> extends AbstractLearner<Instance, Out, DecisionTree<Out>> {

    protected ClassSet classSet;
    protected Array<Out> classes = null;

    protected Learner(Array<Out> classes, Properties parameters, ClassSet classSet) {
      super(parameters);
      this.classSet = classSet;
      this.classes = classes;
    }

    public Learner(Properties parameters) {
      super(parameters);
    }

    public Learner() {
      set(SPLITTER, RandomSplitter.all());
      set(MIN_LEAF_SIZE, 1.0);
    }

    @Override
    public DecisionTree<Out> fit(Input<? extends Instance> in, Output<? extends Out> out) {
      Check.argument(Dataset.isDataset(in), "requires a dataset");
      ClassSet classSet = this.classSet;
      Array<Out> classes = this.classes != null ? this.classes : Outputs.unique(out);
      if (classSet == null) {
        classSet = new ClassSet(out, classes);
      }

      Params p = new Params();
      TreeNode<Instance, ValueThreshold> node = build(in, out, p, classSet);
      return new DecisionTree<>(classes, p.depth, new SimplePredictionVisitor(node));
    }

    protected TreeNode<Instance, ValueThreshold> build(Input<? extends Instance> frame,
        Output<?> target, Params p, ClassSet classSet) {
      return build(frame, target, p, classSet, 1);
    }

    protected TreeNode<Instance, ValueThreshold> build(Input<? extends Instance> frame,
        Output<?> target, Params p, ClassSet classSet, int depth) {
      if (classSet.getTotalWeight() <= getOrDefault(MIN_LEAF_SIZE)
          || depth >= getOrDefault(MAX_DEPTH) || classSet.getTargetCount() == 1) {
        p.depth = Math.max(p.depth, depth);
        return TreeLeaf.fromExamples(classSet);
      }
      TreeSplit<ValueThreshold> maxSplit = get(SPLITTER).find(classSet, frame, target);
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
          TreeNode<Instance, ValueThreshold> leftNode = build(frame, target, p, left, depth + 1);
          TreeNode<Instance, ValueThreshold> rightNode = build(frame, target, p, right, depth + 1);
          return new TreeBranch<>(leftNode, rightNode, classes, maxSplit.getThreshold(), 1);
        }
      }
    }

    private final class Params {
      int depth = 0;
    }
  }

  private static final class SimplePredictionVisitor
      implements TreeVisitor<Instance, ValueThreshold> {

    private static final int MISSING = 0, LEFT = -1, RIGHT = 1;
    private final TreeNode<Instance, ValueThreshold> root;

    SimplePredictionVisitor(TreeNode<Instance, ValueThreshold> node) {
      this.root = node;
    }

    @Override
    public TreeNode<Instance, ValueThreshold> getRoot() {
      return root;
    }

    @Override
    public DoubleArray visitLeaf(TreeLeaf<Instance, ValueThreshold> leaf, Instance example) {
      return leaf.getProbabilities();
    }

    @Override
    public DoubleArray visitBranch(TreeBranch<Instance, ValueThreshold> node, Instance example) {
      Object threshold = node.getThreshold().getValue();
      int axis = node.getThreshold().getAxis();
      int direction = MISSING;
      if (!Is.NA(example.get(axis))) {
        if (Is.nominal(threshold)) {
          direction = Is.equal(example.get(axis), threshold) ? LEFT : RIGHT;
        } else {
          double left = example.getDouble(axis);
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
          return visit(node.getLeft(), example);
      }
    }
  }
}
