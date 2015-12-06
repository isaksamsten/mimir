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
