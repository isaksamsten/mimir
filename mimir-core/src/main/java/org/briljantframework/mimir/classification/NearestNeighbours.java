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
package org.briljantframework.mimir.classification;

import java.util.HashSet;
import java.util.List;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.Property;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.distance.Distance;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric
 * method used for classification and regression.[1] In both cases, the input consists of the k
 * closest training examples in the feature space. The output depends on whether k-NN is used for
 * classification or regression:
 *
 * <p/>
 * In k-NN classification, the output is a class membership. An object is classified by a majority
 * vote of its neighbors, with the object being assigned to the class most common among its k
 * nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply
 * assigned to the class of that single nearest neighbor. In k-NN regression, the output is the
 * property value for the object. This value is the average of the getPosteriorProbabilities of its
 * k nearest neighbors. k-NN is a type of instance-based learning, or lazy learning, where the
 * function is only approximated locally and all computation is deferred until classification. The
 * k-NN algorithm is among the simplest of all machine learning algorithms.
 *
 * @author Isak Karlsson
 */
public class NearestNeighbours<In, Out> extends AbstractClassifier<In, Out>
    implements ProbabilityEstimator<In, Out> {

  public static final Property<Integer> NEIGHBORS = Property.of("neighbours", Integer.class, 1);

  private final Input<In> x;
  private final List<Out> y;
  private final Distance<? super In> distance;

  private final int k;

  private NearestNeighbours(Array<Out> classes, Input<In> x, List<Out> y,
      Distance<? super In> distance, int k) {
    super(classes);
    this.x = x;
    this.y = y;
    this.distance = distance;
    this.k = k;
  }

  @Override
  public DoubleArray estimate(In record) {
    Check.argument(x.getSchema().isValid(record), "illegal instance");

    // Only 1nn
    // TODO: 3/14/16 fix k
    Object cls = null;
    double bestSoFar = Double.POSITIVE_INFINITY;
    for (int i = 0; i < x.size(); i++) {
      double distance = this.distance.compute(x.get(i), record);
      if (distance < bestSoFar) {
        cls = y.get(i);
        bestSoFar = distance;
      }
    }

    Array<?> classes = getClasses();
    DoubleArray estimate = DoubleArray.zeros(classes.size());
    for (int i = 0; i < classes.size(); i++) {
      estimate.set(i, Is.equal(classes.get(i), cls) ? 1 : 0);
    }
    return estimate;
  }

  public DoubleArray pairwiseDistance(Input<? extends In> x) {
    int n = x.size();
    int m = this.x.size();
    DoubleArray distances = DoubleArray.zeros(n, m);
    for (int i = 0; i < n; i++) {
      In ri = x.get(i);
      for (int j = 0; j < m; j++) {
        distances.set(i, j, this.distance.compute(ri, this.x.get(j)));
      }
    }
    return distances;
  }

  /**
   * Computes the distance of the given example to all examples in the search space represented by
   * this classifier
   *
   * @param example the given example
   * @return a {@code [search space size]} array of distances to the given example
   */
  public DoubleArray distance(In example) {
    int n = x.size();
    DoubleArray distances = DoubleArray.zeros(n);
    for (int i = 0; i < n; i++) {
      distances.set(i, this.distance.compute(example, x.get(i)));
    }
    return distances;
  }

  public List<?> getTarget() {
    return y;
  }

  public static class Learner<In, Out>
      extends Predictor.Learner<In, Out, NearestNeighbours<In, Out>> {
    private final Distance<? super In> distance;

    public Learner(int k, Distance<? super In> distance) {
      set(NEIGHBORS, k);
      this.distance = distance;
    }

    @Override
    public NearestNeighbours<In, Out> fit(Input<In> x, List<Out> y) {
      Check.argument(x.size() == y.size(), "The size of x and y don't match: %s != %s.", x.size(),
          y.size());
      return new NearestNeighbours<>(Array.copyOf(new HashSet<>(y)), x, y, distance,
          get(NEIGHBORS));
    }

    @Override
    public String toString() {
      return "k-Nearest Neighbors";
    }
  }
}
