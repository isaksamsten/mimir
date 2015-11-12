package org.briljantframework.classifier.conformal;

import java.util.stream.IntStream;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.classification.NearestNeighbours;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.distance.Distance;
import org.briljantframework.distance.Euclidean;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DistanceNonconformity implements Nonconformity {

  private final NearestNeighbours predictor;
  private final int k;

  public DistanceNonconformity(NearestNeighbours predictor, int k) {
    this.predictor = predictor;
    this.k = k;
  }

  @Override
  public double estimate(Vector example, Object label) {
    Vector labels = predictor.getTarget();
    DoubleArray distances = predictor.distance(example);
    IntArray order = Arrays.order(distances);
    double posDist = 0;
    double negDist = 0;
    int kp = 0;
    int kn = 0;
    for (int i = 0; i < order.size() && (kp < k || kn < k); i++) {
      int o = order.get(i);
      double distance = distances.get(o);
      if (Is.equal(labels.loc().get(o), label) && kp < k) {
        posDist += distance;
        kp++;
      } else if (!Is.equal(labels.loc().get(o), label) && kn < k) {
        negDist += distance;
        kn++;
      }
    }

    if (Double.isNaN(posDist)) {
      return Double.POSITIVE_INFINITY;
    } else if (Double.isNaN(negDist)) {
      return Double.NEGATIVE_INFINITY;
    }
    return negDist == 0 ? 0 : (posDist / negDist) / example.size();
  }

  @Override
  public DoubleArray estimate(DataFrame x, Vector y) {
    DoubleArray array = Arrays.newDoubleArray(x.rows());
    // Run in parallel
    IntStream.range(0, x.rows()).parallel()
        .forEach(i -> array.set(i, estimate(x.loc().getRecord(i), y.loc().get(i))));
    return array;
  }

  /**
   * A nonconformity learner that produces a nonconformity scorer based on the {@code k} nearest
   * neighbours according to the specified {@linkplain Distance distance function}.
   *
   * <p/>
   * The nonconformity score is computed by taking the difference of the distance of the {@code k}
   * closest neighbours of the specified label and the distance of the {@code k} closest neighbours of
   * instances with a different label.
   *
   * <h3>References</h3>
   * <ul>
   * <li>Vovk, V., Gammerman, A., Shafer, G. (2005) Algorithmic Learning in a Random World. New York:
   * Springer. See Chapter 3, p. 54.</li>
   * </ul>
   *
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner implements Nonconformity.Learner {

    private final NearestNeighbours.Learner nearestNeighbors;
    private final int k;

    public Learner(int k) {
      this(k, Euclidean.getInstance());
    }

    public Learner(int k, Distance distance) {
      this.nearestNeighbors = new NearestNeighbours.Learner(k, distance);
      this.k = k;
    }

    @Override
    public Nonconformity fit(DataFrame x, Vector y) {
      return new DistanceNonconformity(nearestNeighbors.fit(x, y), k);
    }

  }
}
