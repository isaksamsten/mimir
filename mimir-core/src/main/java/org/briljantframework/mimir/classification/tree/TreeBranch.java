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
package org.briljantframework.mimir.classification.tree;

import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.series.Series;

/**
 * Created by isak on 2/11/15.
 */
public final class TreeBranch<In> implements TreeNode<In> {

  private final TreeNode<In> left;
  private final TreeNode<In> right;
  private final TreeNode<In> missing;
  private final Array<?> domain;
  private final Series classDistribution;
  private final TreeNodeTest<In> threshold;
  private final double weight;
  private final double impurity;
  private final ClassSet classSet;


  public TreeBranch(TreeNode<In> left, TreeNode<In> right, Array<?> domain,
      TreeNodeTest<In> threshold, double weight, double impurity) {
    this(left, right, null, domain, null, threshold, weight, impurity, null);
  }

  public TreeBranch(TreeNode<In> left, TreeNode<In> right, TreeNode<In> missing, Array<?> domain,
      TreeNodeTest<In> threshold, double weight, double impurity) {
    this(left, right, missing, domain, null, threshold, weight, impurity, null);
  }

  public TreeBranch(TreeNode<In> left, TreeNode<In> right, TreeNode<In> missing, Array<?> domain,
      Series classDistribution, TreeNodeTest<In> threshold, double weight, double impurity, ClassSet classSet) {
    this.left = left;
    this.right = right;
    this.domain = domain;
    this.classDistribution = classDistribution;
    this.threshold = threshold;
    this.weight = weight;
    this.missing = missing;
    this.impurity = impurity;
    this.classSet = classSet;
  }

  public double getImpurity() {
    return impurity;
  }

  public ClassSet getClassSet() {
    return classSet;
  }

  public TreeNode<In> getLeft() {
    return left;
  }

  public TreeNode<In> getRight() {
    return right;
  }

  /**
   * @return null if missing is missing
   */
  public TreeNode<In> getMissing() {
    return missing;
  }

  public TreeNodeTest<In> getTreeNodeTest() {
    return threshold;
  }

  @Override
  public double getWeight() {
    return weight;
  }

  @Override
  public Array<?> getDomain() {
    return domain;
  }

  public Series getClassDistribution() {
    return classDistribution;
  }

  @Override
  public DoubleArray visit(TreeVisitor<In> visitor, In example) {
    return visitor.visitBranch(this, example);
  }
}
