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

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class Ensemble<In> extends AbstractClassifier<In> {

  private final List<? extends Classifier<In>> members;
  private final BooleanArray oobIndicator;

  protected Ensemble(List<?> classes, List<? extends Classifier<In>> members,
      BooleanArray oobIndicator) {
    super(classes);
    this.members = members;
    this.oobIndicator = oobIndicator;
  }

  public static <In> DoubleArray oobEstimates(Ensemble<In> ensemble, Input<? extends In> x) {
    BooleanArray ind = ensemble.getOobIndicator();
    List<Classifier<In>> members = ensemble.getEnsembleMembers();
    DoubleArray estimates = DoubleArray.zeros(x.size(), ensemble.getClasses().size());
    for (int i = 0; i < x.size(); i++) {
      In example = x.get(i);
      DoubleArray estimate = estimates.getRow(i);
      BooleanArray oob = ind.getRow(i);
      int size = 0;
      for (int j = 0; j < oob.size(); j++) {
        if (oob.get(j)) {
          estimate.plusAssign(members.get(j).estimate(example));
          size++;
        }
      }
      estimate.divAssign(size);
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

  public List<Classifier<In>> getEnsembleMembers() {
    return Collections.unmodifiableList(members);
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  @Override
  public DoubleArray estimate(In record) {
    List<DoubleArray> predictions =
        members.parallelStream().map(model -> model.estimate(record)).collect(Collectors.toList());

    int estimators = getEnsembleMembers().size();
    List<?> classes = getClasses();
    DoubleArray m = DoubleArray.zeros(classes.size());
    for (DoubleArray prediction : predictions) {
      m.combineAssign(prediction, (t, o) -> t + o / estimators);
    }
    return m;
  }

  public interface BaseLearner<In, T extends Classifier<In>> {
    Predictor.Learner<In, Object, ? extends T> getLearner(ClassSet set, List<?> classes);
  }

  /**
   * @author Isak Karlsson
   */
  public abstract static class Learner<In, P extends Ensemble<In>>
      implements Predictor.Learner<In, Object, P> {

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

    protected final int size;

    protected Learner(int size) {
      this.size = size;
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
    protected static <In, T extends Classifier<In>> List<T> execute(
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

    /**
     * Get the number of members in the ensemble
     *
     * @return the size of the ensemble
     */
    public int size() {
      return size;
    }
  }
}
