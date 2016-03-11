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

import java.util.List;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * Created by isak on 2/11/15.
 */
public final class TreeBranch<In, T> implements TreeNode<In, T> {

  private final TreeNode<In, T> left;
  private final TreeNode<In, T> right;
  private final TreeNode<In, T> missing;
  private final List<?> domain;
  private final Vector classDistribution;
  private final T threshold;
  private final double weight;


  public TreeBranch(TreeNode<In, T> left, TreeNode<In, T> right, List<?> domain, T threshold,
      double weight) {
    this(left, right, null, domain, null, threshold, weight);
  }

  public TreeBranch(TreeNode<In, T> left, TreeNode<In, T> right, TreeNode<In, T> missing,
      List<?> domain, T threshold, double weight) {
    this(left, right, missing, domain, null, threshold, weight);
  }

  public TreeBranch(TreeNode<In, T> left, TreeNode<In, T> right, TreeNode<In, T> missing,
      List<?> domain, Vector classDistribution, T threshold, double weight) {
    this.left = left;
    this.right = right;
    this.domain = domain;
    this.classDistribution = classDistribution;
    this.threshold = threshold;
    this.weight = weight;
    this.missing = missing;
  }

  public TreeNode<In, T> getLeft() {
    return left;
  }

  public TreeNode<In, T> getRight() {
    return right;
  }

  /**
   * @return null if missing is missing
   */
  public TreeNode<In, T> getMissing() {
    return missing;
  }

  public T getThreshold() {
    return threshold;
  }

  @Override
  public double getWeight() {
    return weight;
  }

  @Override
  public List<?> getDomain() {
    return domain;
  }

  public Vector getClassDistribution() {
    return classDistribution;
  }

  @Override
  public DoubleArray visit(TreeVisitor<In, T> visitor, In example) {
    return visitor.visitBranch(this, example);
  }
}
