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

import java.util.HashSet;
import java.util.List;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.mimir.Properties;
import org.briljantframework.mimir.Property;
import org.briljantframework.mimir.classification.tree.*;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Schema;
import org.briljantframework.mimir.supervised.Predictor;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DecisionTree<Out> extends TreeClassifier<Instance, Out> {

  private static Class<? extends Splitter> SPLITTER_CLASS = Splitter.class;
  public static Property<Double> MIN_LEAF_SIZE = Property.of("min_leaf_size", Double.class, 1.0);

  @SuppressWarnings("unchecked")
  public static Property<Splitter<Instance>> SPLITTER =
      Property.of("splitter", (Class<Splitter<Instance>>) SPLITTER_CLASS);

  public static Property<Integer> MAX_DEPTH =
      Property.of("max_depth", Integer.class, Integer.MAX_VALUE, i -> i > 0);


  private DecisionTree(Array<Out> classes, Schema<Instance> schema, TreeNode<Instance> root,
      TreeVisitor<Instance> predictionVisitor) {
    super(schema, classes, root, predictionVisitor);
  }

  public int getDepth() {
    throw new UnsupportedOperationException();
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner<Out> extends Predictor.Learner<Instance, Out, DecisionTree<Out>> {

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
      set(SPLITTER, RandomSplitter.sqrt());
      set(MIN_LEAF_SIZE, 1.0);
    }

    @Override
    @SuppressWarnings("unchecked")
    public DecisionTree<Out> fit(Input<Instance> in, List<Out> out) {
      Check.argument(in.getSchema() instanceof MultidimensionalSchema, "illegal schema");
      ClassSet classSet = this.classSet;
      Array<Out> classes =
          this.classes != null ? this.classes : Array.copyOf(new HashSet<Out>(out));
      if (classSet == null) {
        classSet = new ClassSet(out, classes);
      }

      TreeNode<Instance> node = build(in, out, classSet);
      return new DecisionTree<>(classes, in.getSchema(), node, new TreeVisitor<>());
    }

    protected TreeNode<Instance> build(Input<? extends Instance> frame, List<?> target,
        ClassSet classSet) {
      return build(frame, target, classSet, 1);
    }

    protected TreeNode<Instance> build(Input<? extends Instance> frame, List<?> target,
        ClassSet classSet, int depth) {
      if (classSet.getTotalWeight() <= getOrDefault(MIN_LEAF_SIZE)
          || depth >= getOrDefault(MAX_DEPTH) || classSet.getTargetCount() == 1) {
        return TreeLeaf.fromExamples(classSet);
      }
      Splitter<Instance> instanceSplitter = get(SPLITTER);
      TreeSplit<Instance> maxSplit = instanceSplitter.find(classSet, frame, target);
      if (maxSplit == null) {
        return TreeLeaf.fromExamples(classSet);
      } else {
        ClassSet left = maxSplit.getLeft();
        ClassSet right = maxSplit.getRight();
        if (left.isEmpty()) {
          return TreeLeaf.fromExamples(right);
        } else if (right.isEmpty()) {
          return TreeLeaf.fromExamples(left);
        } else {
          TreeNode<Instance> leftNode = build(frame, target, left, depth + 1);
          TreeNode<Instance> rightNode = build(frame, target, right, depth + 1);
          return new TreeBranch<>(leftNode, rightNode, classes, maxSplit.getThreshold(), 1,
              maxSplit.getImpurity());
        }
      }
    }

  }

  // private static final class SimplePredictionVisitor extends TreeVisitor<Instance> {
  //
  // private final TreeNode<Instance> root;
  //
  // SimplePredictionVisitor(TreeNode<Instance> node) {
  // super(tester, root);
  // this.root = node;
  // }
  //
  // @Override
  // public TreeNode<Instance> getRoot() {
  // return root;
  // }
  //
  // @Override
  // public DoubleArray visitLeaf(TreeLeaf<Instance> leaf, Instance example) {
  // return leaf.getProbabilities();
  // }
  //
  // @Override
  // public DoubleArray visitBranch(TreeBranch<Instance> node, Instance example) {
  // switch (node.getTreeNodeTest().test(example)) {
  // case LEFT:
  // return visit(node.getLeft(), example);
  // case RIGHT:
  // return visit(node.getRight(), example);
  // case MISSING:
  // default:
  // return visit(node.getLeft(), example);
  // }
  // }
  // }
}
