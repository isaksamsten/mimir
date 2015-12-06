package org.briljantframework.mimir.classification;

import java.util.Collections;
import java.util.Set;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
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

  private DecisionTree(Vector classes, TreeNode<ValueThreshold> node,
      TreeVisitor<ValueThreshold> predictionVisitor) {
    super(classes, node, predictionVisitor);
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

      TreeNode<ValueThreshold> node = build(x, y, classSet);
      return new DecisionTree(classes, node, new SimplePredictionVisitor());
    }

    protected TreeNode<ValueThreshold> build(DataFrame frame, Vector target, ClassSet classSet) {
      return build(frame, target, classSet, 0);
    }

    protected TreeNode<ValueThreshold> build(DataFrame frame, Vector target, ClassSet classSet,
        int depth) {
      if (classSet.getTotalWeight() <= mininumWeight || classSet.getTargetCount() == 1) {
        return TreeLeaf.fromExamples(classSet);
      }
      TreeSplit<ValueThreshold> maxSplit = splitter.find(classSet, frame, target);
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
          TreeNode<ValueThreshold> leftNode = build(frame, target, left, depth + 1);
          TreeNode<ValueThreshold> rightNode = build(frame, target, right, depth + 1);
          return new TreeBranch<>(leftNode, rightNode, classes, maxSplit.getThreshold(), 1);
        }
      }
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
          @SuppressWarnings("unchecked")
          Comparable<Object> leftComparable = example.loc().get(Comparable.class, axis);
          direction = leftComparable.compareTo(threshold) <= 0 ? LEFT : RIGHT;
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
