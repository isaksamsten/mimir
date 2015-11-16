package org.mimirframework.classification;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.mimirframework.classification.tree.ClassSet;
import org.mimirframework.classification.tree.Example;
import org.mimirframework.classification.tree.Gain;
import org.mimirframework.classification.tree.HyperPlaneThreshold;
import org.mimirframework.classification.tree.TreeBranch;
import org.mimirframework.classification.tree.TreeClassifier;
import org.mimirframework.classification.tree.TreeLeaf;
import org.mimirframework.classification.tree.TreeNode;
import org.mimirframework.classification.tree.TreeSplit;
import org.mimirframework.classification.tree.TreeVisitor;
import org.mimirframework.supervised.Predictor;

/**
 * Created by isak on 11/16/15.
 */
public class HyperPlaneTree extends TreeClassifier<HyperPlaneThreshold> {
  protected HyperPlaneTree(Vector classes, TreeNode<HyperPlaneThreshold> node,
      TreeVisitor<HyperPlaneThreshold> predictionVisitor) {
    super(classes, node, predictionVisitor);
  }

  private static class HyperPlaneTreeVisitor implements TreeVisitor<HyperPlaneThreshold> {
    @Override
    public DoubleArray visitLeaf(TreeLeaf<HyperPlaneThreshold> leaf, Vector example) {
      return leaf.getProbabilities();
    }

    @Override
    public DoubleArray visitBranch(TreeBranch<HyperPlaneThreshold> node, Vector example) {
      DoubleArray row = example.toDoubleArray();
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

    public Learner(ClassSet set, Vector classes) {
      this.set = set;
      this.classes = classes;
    }

    @Override
    public HyperPlaneTree fit(DataFrame x, Vector y) {
      ClassSet set = this.set;
      Vector classes = this.classes == null ? Vectors.unique(y) : this.classes;
      if (set == null) {
        set = new ClassSet(y, classes);
      }

      TreeNode<HyperPlaneThreshold> root = build(x.toDoubleArray(), y, set);
      return new HyperPlaneTree(classes, root, new HyperPlaneTreeVisitor());
    }

    private TreeNode<HyperPlaneThreshold> build(DoubleArray x, Vector y, ClassSet set) {
      if (set.getTotalWeight() <= 1.0 || set.getTargetCount() == 1) {
        return TreeLeaf.fromExamples(set);
      }
      TreeSplit<HyperPlaneThreshold> maxSplit = find(set, x, y);
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

    private TreeSplit<HyperPlaneThreshold> find(ClassSet set, DoubleArray x, Vector y) {

      TreeSplit<HyperPlaneThreshold> bestSplit = null;
      double bestImpurity = Double.POSITIVE_INFINITY;
      for (int i = 0; i < 10; i++) {
        int index = set.getRandomSample().getRandomExample().getIndex();
        DoubleArray row = x.getRow(index);
        DoubleArray weights = Arrays.randn(row.size());
        TreeSplit<HyperPlaneThreshold> split =
            split(x, set, weights, null, Arrays.inner(row, weights));
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
  }
}
