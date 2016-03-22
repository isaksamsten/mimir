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
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.classification.tree.RandomSplitter;
import org.briljantframework.mimir.classification.tree.Splitter;
import org.briljantframework.mimir.data.*;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class RandomForest extends Ensemble<Instance> {

  private RandomForest(List<?> classes, List<? extends Classifier<Instance>> members,
      BooleanArray oobIndicator) {
    super(classes, members, oobIndicator);
  }

  @Override
  public String toString() {
    return "Random forest";
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner extends Ensemble.Learner<Instance, RandomForest> {

    private final BaseLearner<Instance, ? extends Classifier<Instance>> learnStrategy;

    public Learner(int size) {
      this(RandomSplitter.withMaximumFeatures(-1).create(), size);
    }

    private Learner(BaseLearner<Instance, ? extends Classifier<Instance>> baseLearner, int size) {
      super(size);
      this.learnStrategy = baseLearner;
    }

    private Learner(Splitter splitter, int size) {
      super(size);
      learnStrategy = (BaseLearner<Instance, Classifier<Instance>>) (set,
          classes) -> new DecisionTree.Learner(splitter, set, classes);
    }

    @Override
    public RandomForest fit(Input<? extends Instance> x, Output<?> y) {
      PropertyPreconditions.checkProperties(getRequiredInputProperties(), x);
      Check.argument(x.size() == y.size());

      List<?> classes = Outputs.unique(y);
      Check.argument(classes.size() > 1, "require more than 1 output.");

      ClassSet classSet = new ClassSet(y, classes);
      List<FitTask> fitTasks = new ArrayList<>();
      BooleanArray oobIndicator = Arrays.booleanArray(x.size(), size());
      for (int i = 0; i < size(); i++) {
        fitTasks
            .add(new FitTask(classSet, x, y, learnStrategy, classes, oobIndicator.getColumn(i)));
      }
      try {
        return new RandomForest(classes, execute(fitTasks), oobIndicator);
      } catch (Exception e) {
        e.printStackTrace();
        return null;
      }
    }

    @Override
    public Collection<Property<?>> getRequiredInputProperties() {
      return java.util.Arrays.asList(Dataset.FEATURE_SIZE, Dataset.FEATURE_TYPES);
    }

    @Override
    public String toString() {
      return "Random Classification Forest";
    }

    private static final class FitTask implements Callable<Classifier<Instance>> {

      private final ClassSet classSet;
      private final Input<? extends Instance> x;
      private final Output<?> y;
      private final List<?> classes;
      private final BooleanArray oobIndicator;
      private final BaseLearner<Instance, ? extends Classifier<Instance>> baseLearner;

      private FitTask(ClassSet classSet, Input<? extends Instance> x, Output<?> y,
          BaseLearner<Instance, ? extends Classifier<Instance>> baseLearner, List<?> classes,
          BooleanArray oobIndicator) {
        this.classSet = classSet;
        this.x = x;
        this.y = y;
        this.baseLearner = baseLearner;
        this.classes = classes;
        this.oobIndicator = oobIndicator;
      }

      @Override
      public Classifier<Instance> call() throws Exception {
        Random random = new Random(Thread.currentThread().getId() * System.currentTimeMillis());
        ClassSet bootstrap = sample(classSet, random);
        return baseLearner.getLearner(bootstrap, classes).fit(x, y);
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

  public static class Configurator implements Classifier.Configurator<Instance, Object, Learner> {

    private RandomSplitter.Builder splitter = RandomSplitter.withMaximumFeatures(-1);
    private int size = 100;
    private BaseLearner<Instance, ? extends Classifier<Instance>> learner = null;

    public Configurator(int size) {
      this.size = size;
    }

    public Configurator setSize(int size) {
      this.size = size;
      return this;
    }

    public Configurator setMaximumFeatures(int size) {
      splitter.setMaximumFeatures(size);
      return this;
    }

    public Configurator setBaseLearner(
        BaseLearner<Instance, ? extends Classifier<Instance>> learner) {
      this.learner = learner;
      return this;
    }

    @Override
    public Learner configure() {
      if (learner == null) {
        return new Learner(splitter.create(), size);
      } else {
        return new Learner(learner, size);
      }
    }
  }
}
