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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.classification.tree.RandomSplitter;
import org.briljantframework.mimir.data.*;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class RandomForest<Out> extends Ensemble<Instance, Out> {

  private final List<?> featureNames;
  private final List<Class<?>> featureTypes;

  private RandomForest(Array<Out> classes, List<?> featureNames, List<Class<?>> featureTypes,
      List<? extends DecisionTree<Out>> members, BooleanArray oobIndicator) {
    super(classes, members, oobIndicator);
    this.featureNames = featureNames;
    this.featureTypes = featureTypes;
  }

  @Override
  public String toString() {
    return "Random forest";
  }

  @Override
  public DoubleArray estimate(Instance input) {
    Check.argument(input.size() == featureNames.size(), "illegal input");
    return averageProbabilities(input);
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner<Out> extends Ensemble.Learner<Instance, Out, RandomForest<Out>> {

    public Learner(int size) {
      super(size);
      set(DecisionTree.SPLITTER, RandomSplitter.sqrt());
    }

    @Override
    public RandomForest<Out> fit(Input<? extends Instance> x, Output<? extends Out> y) {
      Check.argument(Dataset.isDataset(x), "dataset is required");
      Check.argument(x.size() == y.size(), "input and output must have the same size");

      Array<Out> classes = Outputs.unique(y);
      Check.argument(classes.size() > 1, "require more than 1 output.");

      ClassSet classSet = new ClassSet(y, classes);
      int size = get(SIZE);
      BooleanArray oobIndicator = Arrays.booleanArray(x.size(), size);
      List<FitTask<Out>> fitTasks = new ArrayList<>(size);
      for (int i = 0; i < size; i++) {
        fitTasks.add(
            new FitTask<>(classSet, getParameters(), x, y, classes, oobIndicator.getColumn(i)));
      }
      try {
        return new RandomForest<Out>(classes, x.getProperty(Dataset.FEATURE_NAMES),
            x.getProperty(Dataset.FEATURE_TYPES), execute(fitTasks), oobIndicator);
      } catch (Exception e) {
        e.printStackTrace();
        return null;
      }
    }

    @Override
    public String toString() {
      return "Random Classification Forest";
    }

    private static final class FitTask<Out> implements Callable<DecisionTree<Out>> {

      private final ClassSet classSet;
      private final Input<? extends Instance> x;
      private final Output<? extends Out> y;
      private final Array<Out> classes;
      private final BooleanArray oobIndicator;
      private final Properties properties;

      private FitTask(ClassSet classSet, Properties properties, Input<? extends Instance> x,
          Output<? extends Out> y, Array<Out> classes, BooleanArray oobIndicator) {
        this.classSet = classSet;
        this.x = x;
        this.y = y;
        this.classes = classes;
        this.oobIndicator = oobIndicator;
        this.properties = properties;
      }

      @Override
      public DecisionTree<Out> call() throws Exception {
        Random random = new Random(Thread.currentThread().getId() * System.currentTimeMillis());
        ClassSet bootstrap = sample(classSet, random);
        DecisionTree.Learner<Out> learner =
            new DecisionTree.Learner<>(classes, properties, bootstrap);
        return learner.fit(x, y);
      }

      public ClassSet sample(ClassSet classSet, Random random) {
        ClassSet inBag = new ClassSet(classSet.getDomain());
        int[] bootstrap = bootstrap(classSet, random);
        for (ClassSet.Sample sample : classSet.samples()) {
          ClassSet.Sample inSample = ClassSet.Sample.create(sample.getTarget());
          for (Example example : sample) {
            int id = example.getIndex();
            if (bootstrap[id] > 0) {
              inSample.add(example.updateWeight(bootstrap[id]));
            } else {
              oobIndicator.set(id, true);
            }
          }
          if (!inSample.isEmpty()) {
            inBag.add(inSample);
          }
        }
        return inBag;
      }

      private int[] bootstrap(ClassSet sample, Random random) {
        int[] bootstrap = new int[sample.size()];
        for (int i = 0; i < bootstrap.length; i++) {
          int idx = random.nextInt(bootstrap.length);
          bootstrap[idx]++;
        }

        return bootstrap;
      }
    }
  }

}
