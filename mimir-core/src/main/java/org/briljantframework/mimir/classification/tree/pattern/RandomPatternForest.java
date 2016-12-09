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
package org.briljantframework.mimir.classification.tree.pattern;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadLocalRandom;

import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.Ensemble;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.Outputs;
import org.briljantframework.mimir.data.Properties;
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
public class RandomPatternForest<In, Out> extends Ensemble<In, Out> {

  private RandomPatternForest(Array<Out> classes, List<? extends PatternTree<In, Out>> members,
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

  @Override
  public DoubleArray estimate(In input) {
    return averageProbabilities(input);
  }

  public static class DepthEvaluator<In>
      implements org.briljantframework.mimir.evaluation.Evaluator<In, Object> {

    @Override
    public void accept(EvaluationContext<In, Object> ctx) {
      RandomPatternForest<In, Object> predictor =
          (RandomPatternForest<In, Object>) ctx.getPredictor();
      ctx.getMeasureCollection().add("depth", predictor.getAverageDepth());
    }
  }

  private static class TreeFitHelper<In, Out, E> {
    private final PatternFactory<? super In, ? extends E> patternFactory;
    private final PatternDistance<? super In, ? super E> patternDistance;
    private final PatternVisitorFactory<In, E> patternVisitorFactory;

    private TreeFitHelper(PatternFactory<? super In, ? extends E> patternFactory,
        PatternDistance<? super In, ? super E> patternDistance,
        PatternVisitorFactory<In, E> patternVisitorFactory) {
      this.patternFactory = patternFactory;
      this.patternDistance = patternDistance;
      this.patternVisitorFactory = patternVisitorFactory;
    }

    final PatternTree.Learner<In, Out> getPatternTree(Array<Out> classes, ClassSet sample,
        Properties properties) {
      return new PatternTree.Learner<>(classes, patternFactory, patternDistance,
          patternVisitorFactory, sample, properties);
    }
  }

  public static class Learner<In, Out>
      extends Ensemble.Learner<In, Out, RandomPatternForest<In, Out>> {

    // private final PatternTree.Configurator<In, ?> configurator;
//    private final PatternFactory<? super In, ? extends E> patternFactory;
//    private final PatternDistance<? super In, ? super E> patternDistance;
//    private final PatternVisitorFactory<In, E> patternVisitorFactory;

    private final TreeFitHelper<In, Out, ?> treeFitHelper;

    public <E> Learner(PatternFactory<? super In, ? extends E> patternFactory,
        PatternDistance<? super In, ? super E> patternDistance, int size) {
      this(patternFactory, patternDistance, new DefaultPatternVisitorFactory<>(), size);
    }

    public <E> Learner(PatternFactory<? super In, ? extends E> patternFactory,
        PatternDistance<? super In, ? super E> patternDistance,
        PatternVisitorFactory<In, E> patternVisitorFactory, int size) {
      super(size);
//      this.patternFactory = patternFactory;
//      this.patternDistance = patternDistance;
//      this.patternVisitorFactory = patternVisitorFactory;

      this.treeFitHelper = new TreeFitHelper<In, Out, E>(patternFactory, patternDistance, patternVisitorFactory);
    }

    @Override
    public RandomPatternForest<In, Out> fit(Input<? extends In> x, Output<? extends Out> y) {
      Array<Out> classes = Outputs.unique(y);

      ClassSet classSet = new ClassSet(y, classes);
      List<FitTask<In, Out>> tasks = new ArrayList<>();
      int members = get(Ensemble.SIZE);
      BooleanArray oobIndicator = Arrays.booleanArray(x.size(), members);
      for (int i = 0; i < members; i++) {
        BooleanArray oobI = oobIndicator.getColumn(i);
        ClassSet sample = sample(classSet, ThreadLocalRandom.current(), oobI);
        PatternTree.Learner<In, Out> patternTree = treeFitHelper.getPatternTree(classes, sample, getParameters());
        tasks.add(new FitTask<>(x, y, patternTree));
      }

      try {
        List<PatternTree<In, Out>> models = Ensemble.Learner.execute(tasks);
        return new RandomPatternForest<>(classes, models, oobIndicator);
      } catch (Exception e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }

    }

    public ClassSet sample(ClassSet classSet, Random random, BooleanArray oobIndicator) {
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

    @Override
    public String toString() {
      return "Ensemble of Randomized Shapelet Trees";
    }

    private static final class FitTask<In, Out> implements Callable<PatternTree<In, Out>> {
      private final Input<? extends In> x;
      private final Output<? extends Out> y;
      private final PatternTree.Learner<In, Out> patternTree;
      // private final BooleanArray oobIndicator;


      // private FitTask(ClassSet classSet, Input<? extends In> x, Output<?> y,
      // PatternTree.Configurator<In, ?> patternTree, List<?> classes,
      // BooleanArray oobIndicator) {
      // this.classSet = classSet;
      // this.x = x;
      // this.y = y;
      // this.classes = classes;
      // this.patternTree = patternTree;
      // this.oobIndicator = oobIndicator;
      // }

      public FitTask(Input<? extends In> x, Output<? extends Out> y,
          PatternTree.Learner<In, Out> patternTree) {
        this.patternTree = patternTree;
        this.x = x;
        this.y = y;
        // this.oobIndicator = oobIndicator;

      }

      @Override
      public PatternTree<In, Out> call() throws Exception {
        // Random random = new Random(Thread.currentThread().getId() * System.nanoTime());
        // ClassSet sample = sample(classSet, random);
        return patternTree.fit(x, y);
      }

      // public ClassSet sampleNoBootstrap(c)
    }
  }
}
