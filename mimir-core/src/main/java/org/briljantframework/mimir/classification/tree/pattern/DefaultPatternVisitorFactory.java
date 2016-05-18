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
package org.briljantframework.mimir.classification.tree.pattern;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.classification.tree.TreeBranch;
import org.briljantframework.mimir.classification.tree.TreeLeaf;
import org.briljantframework.mimir.classification.tree.TreeNode;
import org.briljantframework.mimir.classification.tree.TreeVisitor;

/**
 * Created by isak on 5/12/16.
 */
public class DefaultPatternVisitorFactory<T, E> implements PatternVisitorFactory<T, E> {
  @Override
  public TreeVisitor<T, PatternTree.Threshold<E>> createVisitor(
      TreeNode<T, PatternTree.Threshold<E>> root,
      PatternDistance<? super T, ? super E> patternDistance) {
    return new TreeVisitor<T, PatternTree.Threshold<E>>() {
      @Override
      public TreeNode<T, PatternTree.Threshold<E>> getRoot() {
        return root;
      }

      @Override
      public DoubleArray visitLeaf(TreeLeaf<T, PatternTree.Threshold<E>> leaf, T example) {
        return leaf.getProbabilities();
      }

      @Override
      public DoubleArray visitBranch(TreeBranch<T, PatternTree.Threshold<E>> node, T example) {
        E shapelet = node.getThreshold().getPattern();
        double threshold = node.getThreshold().getDistance();
        double computedDistance = patternDistance.computeDistance(example, shapelet);
        if (Is.NA(computedDistance)) {
          if (node.getMissing() != null) {
            return visit(node.getMissing(), example);
          }
          return visit(node.getRight(), example);
        } else {
          if (computedDistance < threshold) {
            return visit(node.getLeft(), example);
          } else {
            return visit(node.getRight(), example);
          }
        }
      }
    };
  }

}
