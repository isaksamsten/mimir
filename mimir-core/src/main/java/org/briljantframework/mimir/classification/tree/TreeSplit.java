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

/**
 * Created by isak on 2/11/15.
 */
public final class TreeSplit<E> {

  private final ClassSet right;
  private final ClassSet left;
  private final ClassSet missing;
  private final E threshold;
  private double impurity = 0;

  public TreeSplit(ClassSet left, ClassSet right, ClassSet missing, E threshold) {
    this.left = left;
    this.right = right;
    this.threshold = threshold;
    this.missing = missing;
  }

  public TreeSplit(ClassSet left, ClassSet right, E threshold) {
    this.left = left;
    this.right = right;
    this.threshold = threshold;
    this.missing = null;
  }

  /**
   * @return null if missing is missing
   */
  public ClassSet getMissing() {
    return missing;
  }

  public ClassSet getRight() {
    return right;
  }

  public ClassSet getLeft() {
    return left;
  }

  public double getImpurity() {
    return impurity;
  }

  public void setImpurity(double impurity) {
    this.impurity = impurity;
  }

  public E getThreshold() {
    return threshold;
  }

  public double size() {
    return right.getTotalWeight() + left.getTotalWeight();
  }

  @Override
  public String toString() {
    return String.format("Binary(left=%s, right=%s)", left, right);
  }
}
