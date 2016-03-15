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
package org.briljantframework.mimir.classification.conformal;

import java.util.List;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.classification.NearestNeighbours;
import org.briljantframework.mimir.distance.Distance;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DistanceNonconformity<In> implements ClassifierNonconformity<In> {

  private final NearestNeighbours<In> classifier;
  private final int k;

  public DistanceNonconformity(NearestNeighbours<In> classifier, int k) {
    this.classifier = classifier;
    this.k = k;
  }

  @Override
  public double estimate(In example, Object label) {
    Output<?> labels = classifier.getTarget();
    DoubleArray distances = classifier.distance(example);
    IntArray order = Arrays.order(distances);
    double posDist = 0;
    double negDist = 0;
    int kp = 0;
    int kn = 0;
    for (int i = 0; i < order.size() && (kp < k || kn < k); i++) {
      int o = order.get(i);
      double distance = distances.get(o);
      if (Is.equal(labels.get(o), label) && kp < k) {
        posDist += distance;
        kp++;
      } else if (!Is.equal(labels.get(o), label) && kn < k) {
        negDist += distance;
        kn++;
      }
    }

    if (Double.isNaN(posDist)) {
      return Double.POSITIVE_INFINITY;
    } else if (Double.isNaN(negDist)) {
      return Double.NEGATIVE_INFINITY;
    }
    return negDist == 0 ? 0 : posDist / negDist;
  }

  @Override
  public List<?> getClasses() {
    return classifier.getClasses();
  }

  /**
   * A nonconformity learner that produces a nonconformity scorer based on the {@code k} nearest
   * neighbours according to the specified {@linkplain Distance distance function}.
   *
   * <p/>
   * The nonconformity score is computed by taking the difference of the distance of the {@code k}
   * closest neighbours of the specified label and the distance of the {@code k} closest neighbours
   * of instances with a different label.
   *
   * <h3>References</h3>
   * <ul>
   * <li>Vovk, V., Gammerman, A., Shafer, G. (2005) Algorithmic Learning in a Random World. New
   * York: Springer. See Chapter 3, p. 54.</li>
   * </ul>
   *
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner<In>
      implements ClassifierNonconformity.Learner<In, DistanceNonconformity<In>> {

    private final NearestNeighbours.Learner<In> nearestNeighbors;
    private final int k;

    public Learner(int k, Distance<In> distance) {
      this.nearestNeighbors = new NearestNeighbours.Learner<>(k, distance);
      this.k = k;
    }

    @Override
    public DistanceNonconformity<In> fit(Input<? extends In> x, Output<?> y) {
      return new DistanceNonconformity<>(nearestNeighbors.fit(x, y), k);
    }
  }
}
