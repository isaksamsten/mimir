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

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.Outputs;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.evaluation.EvaluationContext;

/**
 * <h1>Publications</h1>
 * <ul>
 * <li>Karlsson, I., Bostrom, H., Papapetrou, P. Forests of Randomized Shapelet Trees In Proc. the
 * 3rd International Symposium on Learning and Data Sciences (SLDS), 2015</li>
 * </ul>
 *
 * @author Isak Karlsson
 */
public class RandomPatternForest<In> extends Ensemble<In> {

  private RandomPatternForest(List<?> classes, List<? extends Classifier<In>> members,
      BooleanArray oobIndicator) {
    super(classes, members, oobIndicator);
  }

  public double getAverageDepth() {
    double depth = 0;
    for (Classifier classifier : getEnsembleMembers()) {
      if (classifier instanceof PatternTree) {
        int d = ((PatternTree) classifier).getDepth();
        depth += d;
      }
    }
    return depth / getEnsembleMembers().size();
  }

  public static class Configurator<In, E>
      implements Classifier.Configurator<In, Object, Learner<In>> {

    private final PatternTree.Configurator<In, E> shapeletTree;
    private int size = 100;

    public Configurator(PatternTree.PatternFactory<? super In, ? extends E> patternFactory,
        PatternTree.PatternDistance<? super In, ? super E> patternDistance, int size) {
      this.size = size;
      this.shapeletTree = new PatternTree.Configurator<>(patternFactory, patternDistance);
    }

    public Configurator<In, E> setMinimumSplitSize(double minSplitSize) {
      shapeletTree.setMinimumSplit(minSplitSize);
      return this;
    }

    public Configurator<In, E> setPatternDistance(
        PatternTree.PatternDistance<? super In, ? super E> patternDistance) {
      shapeletTree.setPatternDistance(patternDistance);
      return this;
    }

    public Configurator<In, E> setPatternFactory(
        PatternTree.PatternFactory<? super In, ? extends E> patternFactory) {
      shapeletTree.setPatternFactory(patternFactory);
      return this;
    }

    public Configurator<In, E> setPatternCount(int maxShapelets) {
      shapeletTree.setPatternCount(maxShapelets);
      return this;
    }

    public Configurator<In, E> setSize(int size) {
      this.size = size;
      return this;
    }

    public Configurator<In, E> setAssessment(PatternTree.Learner.Assessment assessment) {
      shapeletTree.setAssessment(assessment);
      return this;
    }

    @Override
    public Learner<In> configure() {
      return new Learner<>(shapeletTree, size);
    }
  }

  public static class Evaluator<In> implements
      org.briljantframework.mimir.evaluation.Evaluator<In, Object, RandomPatternForest<In>> {

    @Override
    public void accept(EvaluationContext<? extends In, ?, ? extends RandomPatternForest<In>> ctx) {
      ctx.getMeasureCollection().add("depth", ctx.getPredictor().getAverageDepth());
    }
  }

  public static class Learner<In> extends Ensemble.Learner<In, RandomPatternForest<In>> {

    private final PatternTree.Configurator<In, ?> configurator;

    private Learner(PatternTree.Configurator<In, ?> configurator, int size) {
      super(size);
      this.configurator = configurator;
    }

    public <E> Learner(PatternTree.PatternFactory<In, E> patternFactory,
        PatternTree.PatternDistance<In, E> patternDistance, int size) {
      this(new PatternTree.Configurator<>(patternFactory, patternDistance), size);
    }

    @Override
    public RandomPatternForest<In> fit(Input<? extends In> x, Output<?> y) {
      List<?> classes = Outputs.unique(y);

      ClassSet classSet = new ClassSet(y, classes);
      List<FitTask<In>> tasks = new ArrayList<>();
      BooleanArray oobIndicator = Arrays.booleanArray(x.size(), size());
      for (int i = 0; i < size(); i++) {
        tasks.add(new FitTask<>(classSet, x, y, configurator, classes, oobIndicator.getColumn(i)));
      }

      try {
        List<PatternTree<In>> models = Ensemble.Learner.execute(tasks);
        return new RandomPatternForest<>(classes, models, oobIndicator);
      } catch (Exception e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }

    }

    @Override
    public String toString() {
      return "Ensemble of Randomized Shapelet Trees";
    }

    private static final class FitTask<In> implements Callable<PatternTree<In>> {

      private final ClassSet classSet;
      private final Input<? extends In> x;
      private final Output<?> y;
      private final List<?> classes;
      private final PatternTree.Configurator<In, ?> configurator;
      private final BooleanArray oobIndicator;


      private FitTask(ClassSet classSet, Input<? extends In> x, Output<?> y,
          PatternTree.Configurator<In, ?> configurator, List<?> classes,
          BooleanArray oobIndicator) {
        this.classSet = classSet;
        this.x = x;
        this.y = y;
        this.classes = classes;
        this.configurator = configurator;
        this.oobIndicator = oobIndicator;
      }

      @Override
      public PatternTree<In> call() throws Exception {
        Random random = new Random(Thread.currentThread().getId() * System.nanoTime());
        ClassSet sample = sample(classSet, random);
        return new PatternTree.Learner<>(configurator, sample, classes).fit(x, y);
      }

      // public ClassSet sampleNoBootstrap(c)

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
