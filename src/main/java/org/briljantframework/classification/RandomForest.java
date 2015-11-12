package org.briljantframework.classification;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.classification.tree.ClassSet;
import org.briljantframework.classification.tree.Example;
import org.briljantframework.classification.tree.RandomSplitter;
import org.briljantframework.classification.tree.Splitter;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class RandomForest extends Ensemble {

  private RandomForest(Vector classes, List<? extends Classifier> members, BooleanArray oobIndicator) {
    super(classes, members, oobIndicator);
  }

  @Override
  public String toString() {
    return "Random forest";
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner extends Ensemble.Learner<RandomForest> {

    private final Splitter splitter;

    protected Learner(Splitter splitter, int size) {
      super(size);
      this.splitter = splitter;
    }

    public Learner(int size) {
      super(size);
      splitter = RandomSplitter.withMaximumFeatures(-1).create();
    }

    @Override
    public RandomForest fit(DataFrame x, Vector y) {
      Vector classes = Vectors.unique(y);
      ClassSet classSet = new ClassSet(y, classes);
      List<FitTask> fitTasks = new ArrayList<>();
      BooleanArray oobIndicator = Arrays.newBooleanArray(x.rows(), size());
      for (int i = 0; i < size(); i++) {
        fitTasks.add(new FitTask(classSet, x, y, splitter, classes, oobIndicator.getColumn(i)));
      }
      try {
        return new RandomForest(classes, execute(fitTasks), oobIndicator);
      } catch (Exception e) {
        e.printStackTrace();
        return null;
      }
    }

    @Override
    public String toString() {
      return "Random Classification Forest";
    }

    private static final class FitTask implements Callable<Classifier> {

      private final ClassSet classSet;
      private final DataFrame x;
      private final Vector y;
      private final Splitter splitter;
      private final Vector classes;
      private final BooleanArray oobIndicator;

      private FitTask(ClassSet classSet, DataFrame x, Vector y, Splitter splitter, Vector classes,
          BooleanArray oobIndicator) {
        this.classSet = classSet;
        this.x = x;
        this.y = y;
        this.splitter = splitter;
        this.classes = classes;
        this.oobIndicator = oobIndicator;
      }

      @Override
      public Classifier call() throws Exception {
        Random random = new Random(Thread.currentThread().getId() * System.currentTimeMillis());
        ClassSet bootstrap = sample(classSet, random);
        return new DecisionTree.Learner(splitter, bootstrap, classes).fit(x, y);
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

  public static class Configurator implements Classifier.Configurator<Learner> {

    private RandomSplitter.Builder splitter = RandomSplitter.withMaximumFeatures(-1);
    private int size = 100;

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

    @Override
    public Learner configure() {
      return new Learner(splitter.create(), size);
    }
  }
}
