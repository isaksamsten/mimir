package org.mimirframework.classification;

import java.util.Collections;
import java.util.Set;

import org.briljantframework.array.DoubleArray;
import org.mimirframework.classification.tree.TreeClassifier;
import org.mimirframework.classification.tree.ValueThreshold;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DecisionTree extends TreeClassifier<org.mimirframework.classification.tree.ValueThreshold> {

  private DecisionTree(Vector classes, org.mimirframework.classification.tree.TreeNode<ValueThreshold> node,
      org.mimirframework.classification.tree.TreeVisitor<ValueThreshold> predictionVisitor) {
    super(classes, node, predictionVisitor);
  }

  @Override
  public Set<org.mimirframework.supervised.Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner implements org.mimirframework.supervised.Predictor.Learner<DecisionTree> {

    protected final double mininumWeight = 1;
    protected final org.mimirframework.classification.tree.Splitter splitter;

    protected org.mimirframework.classification.tree.ClassSet classSet;
    protected Vector classes = null;


    public Learner(org.mimirframework.classification.tree.Splitter splitter) {
      this(splitter, null, null);
    }

    protected Learner(org.mimirframework.classification.tree.Splitter splitter, org.mimirframework.classification.tree.ClassSet classSet, Vector classes) {
      this.splitter = splitter;
      this.classSet = classSet;
      this.classes = classes;
    }

    @Override
    public DecisionTree fit(DataFrame x, Vector y) {
      org.mimirframework.classification.tree.ClassSet classSet = this.classSet;
      Vector classes = this.classes != null ? this.classes : Vectors.unique(y);
      if (classSet == null) {
        classSet = new org.mimirframework.classification.tree.ClassSet(y, classes);
      }

      org.mimirframework.classification.tree.TreeNode<ValueThreshold> node = build(x, y, classSet);
      return new DecisionTree(classes, node, new SimplePredictionVisitor());
    }

    protected org.mimirframework.classification.tree.TreeNode<ValueThreshold> build(DataFrame frame, Vector target, org.mimirframework.classification.tree.ClassSet classSet) {
      return build(frame, target, classSet, 0);
    }

    protected org.mimirframework.classification.tree.TreeNode<ValueThreshold> build(DataFrame frame, Vector target, org.mimirframework.classification.tree.ClassSet classSet,
                                                                                    int depth) {
      if (classSet.getTotalWeight() <= mininumWeight || classSet.getTargetCount() == 1) {
        return org.mimirframework.classification.tree.TreeLeaf.fromExamples(classSet);
      }
      org.mimirframework.classification.tree.TreeSplit<ValueThreshold> maxSplit = splitter.find(classSet, frame, target);
      if (maxSplit == null) {
        return org.mimirframework.classification.tree.TreeLeaf.fromExamples(classSet);
      } else {
        org.mimirframework.classification.tree.ClassSet left = maxSplit.getLeft();
        org.mimirframework.classification.tree.ClassSet right = maxSplit.getRight();
        if (left.isEmpty()) {
          return org.mimirframework.classification.tree.TreeLeaf.fromExamples(right);
        } else if (right.isEmpty()) {
          return org.mimirframework.classification.tree.TreeLeaf.fromExamples(left);
        } else {
          org.mimirframework.classification.tree.TreeNode<ValueThreshold> leftNode = build(frame, target, left, depth + 1);
          org.mimirframework.classification.tree.TreeNode<ValueThreshold> rightNode = build(frame, target, right, depth + 1);
          return new org.mimirframework.classification.tree.TreeBranch<>(leftNode, rightNode, classes, maxSplit.getThreshold(), 1);
        }
      }
    }
  }

  private static final class SimplePredictionVisitor implements org.mimirframework.classification.tree.TreeVisitor<ValueThreshold> {

    private static final int MISSING = 0, LEFT = -1, RIGHT = 1;

    @Override
    public DoubleArray visitLeaf(org.mimirframework.classification.tree.TreeLeaf<ValueThreshold> leaf, Vector example) {
      return leaf.getProbabilities();
    }

    @Override
    public DoubleArray visitBranch(org.mimirframework.classification.tree.TreeBranch<ValueThreshold> node, Vector example) {
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
          // direction = example.compare(axis, (Comparable<?>) threshold) <= 0 ? LEFT : RIGHT;
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
