package org.mimirframework.classification;

import java.util.Collections;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.mimirframework.distance.Distance;
import org.mimirframework.distance.EuclideanDistance;
import org.mimirframework.supervised.Characteristic;
import org.mimirframework.supervised.Predictor;

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
public class NearestNeighbours extends AbstractClassifier {

  private final DataFrame x;
  private final Vector y;
  private final Distance distance;
  private final int k;

  private NearestNeighbours(DataFrame x, Vector y, Distance distance, int k, Vector classes) {
    super(classes);
    this.x = x;
    this.y = y;

    this.distance = distance;
    this.k = k;
  }

  public double n() {
    return x.columns();
  }

  @Override
  public DoubleArray estimate(Vector record) {
    // Only 1nn
    Object cls = null;
    double bestSoFar = Double.POSITIVE_INFINITY;
    for (int i = 0; i < x.rows(); i++) {
      double distance = this.distance.compute(x.loc().getRecord(i), record);
      if (distance < bestSoFar) {
        cls = y.loc().get(Object.class, i);
        bestSoFar = distance;
      }
    }

    Vector classes = getClasses();
    DoubleArray estimate = DoubleArray.zeros(classes.size());
    for (int i = 0; i < classes.size(); i++) {
      estimate.set(i, classes.loc().get(Object.class, i).equals(cls) ? 1 : 0);
    }
    return estimate;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  public DoubleArray distance(DataFrame x) {
    int n = x.rows();
    int m = this.x.rows();
    DoubleArray distances = DoubleArray.zeros(n, m);
    for (int i = 0; i < n; i++) {
      Vector ri = x.loc().getRecord(i);
      for (int j = 0; j < m; j++) {
        distances.set(i, j, this.distance.compute(ri, this.x.loc().getRecord(j)));
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
  public DoubleArray distance(Vector example) {
    int n = x.rows();
    DoubleArray distances = DoubleArray.zeros(n);
    for (int i = 0; i < n; i++) {
      distances.set(i, this.distance.compute(example, x.loc().getRecord(i)));
    }
    return distances;
  }

  public Vector getTarget() {
    return y;
  }

  /**
   * A nearest neighbour learner learns a nearest neighbours classifier
   */
  public static class Learner implements Predictor.Learner<NearestNeighbours> {

    private final int neighbors;
    private final Distance distance;

    private Learner(Configurator builder) {
      this.neighbors = builder.neighbors;
      this.distance = builder.distance;
    }

    public Learner(int k) {
      this(k, EuclideanDistance.getInstance());
    }

    public Learner(int k, Distance distance) {
      this.neighbors = k;
      this.distance = distance;
    }

    @Override
    public NearestNeighbours fit(DataFrame x, Vector y) {
      Check.argument(x.rows() == y.size(), "The size of x and y don't match: %s != %s.", x.rows(),
          y.size());
      return new NearestNeighbours(x, y, distance, neighbors, Vectors.unique(y));
    }

    @Override
    public String toString() {
      return "k-Nearest Neighbors";
    }
  }

  public static class Configurator implements Predictor.Configurator<Learner> {

    public int neighbors;
    private Distance distance = EuclideanDistance.getInstance();

    public Configurator(int neighbors) {
      this.neighbors = neighbors;
    }

    public Predictor.Configurator setNeighbors(int k) {
      this.neighbors = k;
      return this;
    }

    public Predictor.Configurator setDistance(Distance distance) {
      this.distance = distance;
      return this;
    }

    public Learner configure() {
      return new Learner(this);
    }
  }
}
