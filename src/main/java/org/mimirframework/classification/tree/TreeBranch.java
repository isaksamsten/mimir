/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.classification.tree;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * Created by isak on 2/11/15.
 */
public final class TreeBranch<T> implements TreeNode<T> {

  private final TreeNode<T> left;
  private final TreeNode<T> right;
  private final Vector domain;
  private final Vector classDistribution;
  private final T threshold;
  private final double weight;

  public TreeBranch(TreeNode<T> left, TreeNode<T> right, Vector domain, T threshold, double weight) {
    this(left, right, domain, null, threshold, weight);
  }

  public TreeBranch(TreeNode<T> left, TreeNode<T> right, Vector domain, Vector classDistribution,
      T threshold, double weight) {
    this.left = left;
    this.right = right;
    this.domain = domain;
    this.classDistribution = classDistribution;
    this.threshold = threshold;
    this.weight = weight;
  }

  public TreeNode<T> getLeft() {
    return left;
  }

  public TreeNode<T> getRight() {
    return right;
  }

  public T getThreshold() {
    return threshold;
  }

  public Vector getClassDistribution() {
    return classDistribution;
  }

  @Override
  public double getWeight() {
    return weight;
  }

  @Override
  public Vector getDomain() {
    return domain;
  }

  @Override
  public DoubleArray visit(TreeVisitor<T> visitor, Vector example) {
    return visitor.visitBranch(this, example);
  }
}
