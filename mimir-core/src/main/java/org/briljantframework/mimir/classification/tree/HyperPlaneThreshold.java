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

import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;

/**
 * @author Isak Karlsson
 */
public class HyperPlaneThreshold {

  private final DoubleArray weights;
  private final IntArray features;
  private final double threshold;

  public HyperPlaneThreshold(DoubleArray weights, IntArray features, double threshold) {
    this.weights = weights;
    this.features = features;
    this.threshold = threshold;
  }

  public IntArray getFeatures() {
    return features;
  }

  public DoubleArray getWeights() {
    return weights;
  }

  public double getThreshold() {
    return threshold;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;

    HyperPlaneThreshold that = (HyperPlaneThreshold) o;

    return !(weights != null ? !weights.equals(that.weights) : that.weights != null)
        && !(features != null ? !features.equals(that.features) : that.features != null);

  }

  @Override
  public int hashCode() {
    int result = weights != null ? weights.hashCode() : 0;
    result = 31 * result + (features != null ? features.hashCode() : 0);
    return result;
  }

  @Override
  public String toString() {
    return "HyperPlaneThreshold{" + "weights=" + weights + ", features=" + features + '}';
  }
}
