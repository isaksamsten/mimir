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

import static org.briljantframework.array.Arrays.div;
import static org.briljantframework.array.Arrays.plus;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Property;
import org.briljantframework.mimir.supervised.AbstractLearner;
import org.briljantframework.mimir.supervised.Characteristic;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public abstract class Ensemble<In, Out> extends AbstractClassifier<In, Out>
    implements ProbabilityEstimator<In, Out> {

  /**
   * The number of members in the ensemble
   */
  public static final Property<Integer> SIZE =
      Property.of("ensemble_size", Integer.class, 100, i -> i > 0);

  private final List<? extends ProbabilityEstimator<In, Out>> members;
  private final BooleanArray oobIndicator;

  protected Ensemble(Array<Out> classes, List<? extends ProbabilityEstimator<In, Out>> members,
      BooleanArray oobIndicator) {
    super(classes);
    this.members = members;
    this.oobIndicator = oobIndicator;
  }

  public static <In> DoubleArray estimateOutOfBagProbabilities(Ensemble<In, ?> ensemble,
      Input<? extends In> x) {
    BooleanArray ind = ensemble.getOobIndicator();
    Check.argument(ind.rows() == x.size(), "input and oob indicator does not match");

    List<? extends ProbabilityEstimator<In, ?>> members = ensemble.getEnsembleMembers();
    DoubleArray estimates = DoubleArray.zeros(x.size(), ensemble.getClasses().size());
    for (int i = 0; i < x.size(); i++) {
      In example = x.get(i);
      DoubleArray estimate = estimates.getRow(i);
      BooleanArray oob = ind.getRow(i);
      int size = 0;
      for (int j = 0; j < oob.size(); j++) {
        if (oob.get(j)) {
          plus(members.get(j).estimate(example), estimate, estimate);
          size++;
        }
      }
      div(DoubleArray.of(size), estimate, estimate);
    }
    return estimates;
  }

  /**
   * Shape = {@code [no training samples, no members]}, if element e<sup>i,j</sup> is {@code true}
   * the i:th training sample is out of the j:th members training sample. Vector
   * 
   * @return the out of bag indicator matrix
   */
  public BooleanArray getOobIndicator() {
    return oobIndicator;
  }

  public List<ProbabilityEstimator<In, Out>> getEnsembleMembers() {
    return Collections.unmodifiableList(members);
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  protected DoubleArray averageProbabilities(In record) {
    List<DoubleArray> predictions =
        members.parallelStream().map(model -> model.estimate(record)).collect(Collectors.toList());

    int estimators = getEnsembleMembers().size();
    Array<?> classes = getClasses();
    DoubleArray m = DoubleArray.zeros(classes.size());
    for (DoubleArray prediction : predictions) {
      m.combineAssign(prediction, (t, o) -> t + o / estimators);
    }
    return m;
  }

  /**
   * @author Isak Karlsson
   */
  public abstract static class Learner<In, Out, P extends Ensemble<In, Out>>
      extends AbstractLearner<In, Out, P> {

    private final static ThreadPoolExecutor THREAD_POOL;
    private final static int CORES;

    static {
      CORES = Runtime.getRuntime().availableProcessors();
      if (CORES <= 1) {
        THREAD_POOL = null;
      } else {
        THREAD_POOL = (ThreadPoolExecutor) Executors.newFixedThreadPool(CORES, r -> {
          Thread thread = new Thread(r);
          thread.setDaemon(true);
          return thread;
        });
      }
    }


    protected Learner(int size) {
      set(SIZE, size);
    }

    /**
     * Executes {@code callable} either sequential or in parallel depending on the number of
     * available cores.
     *
     * @param callables the callables
     * @param <T> the models produced
     * @return a list of produced models
     * @throws Exception if something goes wrong
     */
    protected static <In, Out, T extends ProbabilityEstimator<In, Out>> List<T> execute(
        Collection<? extends Callable<T>> callables) throws Exception {
      List<T> models = new ArrayList<>();
      if (THREAD_POOL != null && THREAD_POOL.getActiveCount() < CORES) {
        for (Future<T> future : THREAD_POOL.invokeAll(callables)) {
          models.add(future.get());
        }
      } else {
        for (Callable<T> callable : callables) {
          models.add(callable.call());
        }
      }
      return models;
    }
  }
}
