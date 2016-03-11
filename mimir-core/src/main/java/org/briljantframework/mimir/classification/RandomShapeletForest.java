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
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Inputs;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.Outputs;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.Example;
import org.briljantframework.mimir.distance.Distance;
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
public class RandomShapeletForest extends Ensemble<Vector> {

  private final DoubleArray lengthImportance;
  private final DoubleArray positionImportance;

  private RandomShapeletForest(List<?> classes, DoubleArray apriori,
      List<? extends Classifier<Vector>> members, DoubleArray lengthImportance,
      DoubleArray positionImportance, BooleanArray oobIndicator) {
    super(classes, members, oobIndicator);
    this.lengthImportance = lengthImportance;
    this.positionImportance = positionImportance;
  }

  public static Configurator withSize(int size) {
    return new Configurator(size);
  }

  public DoubleArray getLengthImportance() {
    return lengthImportance;
  }

  public DoubleArray getPositionImportance() {
    return positionImportance;
  }

  public double getAverageDepth() {
    double depth = 0;
    for (Classifier classifier : getEnsembleMembers()) {
      if (classifier instanceof ShapeletTree) {
        int d = ((ShapeletTree) classifier).getDepth();
        depth += d;
      }
    }
    return depth / getEnsembleMembers().size();
  }

  public static class Configurator implements Classifier.Configurator<Vector, Object, Learner> {

    private final ShapeletTree.Configurator shapeletTree = new ShapeletTree.Configurator();
    private int size = 100;

    public Configurator(int size) {
      this.size = size;
    }

    public Configurator setMinimumSplitSize(double minSplitSize) {
      shapeletTree.setMinimumSplit(minSplitSize);
      return this;
    }

    public Configurator setLowerLength(double lower) {
      shapeletTree.setLowerLength(lower);
      return this;
    }

    public Configurator setUpperLength(double upper) {
      shapeletTree.setUpperLength(upper);
      return this;
    }

    public Configurator setMaximumShapelets(int maxShapelets) {
      shapeletTree.setMaximumShapelets(maxShapelets);
      return this;
    }

    public Configurator setDistance(Distance distance) {
      shapeletTree.setDistance(distance);
      return this;
    }

    public Configurator setCategoricDistance(Distance categoricDistance) {
      shapeletTree.setCategoricDistance(categoricDistance);
      return this;
    }

    public Configurator setSize(int size) {
      this.size = size;
      return this;
    }

    public Configurator setSampleMode(ShapeletTree.Learner.SampleMode sampleMode) {
      shapeletTree.setSampleMode(sampleMode);
      return this;
    }

    public Configurator setAssessment(ShapeletTree.Learner.Assessment assessment) {
      shapeletTree.setAssessment(assessment);
      return this;
    }

    @Override
    public Learner configure() {
      return new Learner(shapeletTree, size);
    }
  }

  public static class Evaluator implements
      org.briljantframework.mimir.evaluation.Evaluator<Vector, Object, RandomShapeletForest> {

    @Override
    public void accept(EvaluationContext<Vector, Object, ? extends RandomShapeletForest> ctx) {
      ctx.getMeasureCollection().add("depth", ctx.getPredictor().getAverageDepth());
    }
  }

  public static class Learner extends Ensemble.Learner<Vector, RandomShapeletForest> {

    private final ShapeletTree.Configurator configurator;

    private Learner(ShapeletTree.Configurator configurator, int size) {
      super(size);
      this.configurator = configurator;
    }

    @Override
    public RandomShapeletForest fit(Input<? extends Vector> x, Output<?> y) {
      List<?> classes = Outputs.unique(y);

      ClassSet classSet = new ClassSet(y, classes);
      List<FitTask> tasks = new ArrayList<>();
      BooleanArray oobIndicator = Arrays.booleanArray(x.size(), size());
      for (int i = 0; i < size(); i++) {
        tasks.add(new FitTask(classSet, x, y, configurator, classes, oobIndicator.getColumn(i)));
      }

      try {
        List<ShapeletTree> models = Ensemble.Learner.execute(tasks);
        int features = Inputs.features(x);
        DoubleArray lenSum = DoubleArray.zeros(features);
        DoubleArray posSum = DoubleArray.zeros(features);
        for (ShapeletTree m : models) {
          lenSum.plusAssign(m.getLengthImportance());
          posSum.plusAssign(m.getPositionImportance());
        }

        // // ShapeletTree.ShapeStore store = new ShapeletTree.ShapeStore();
        // // ShapeletTree.ShapeStore store1 = models.get(0).getStore();
        // // for (int i = 0; i < store1.shapes.size(); i++) {
        // // store.shapes.add(store1.shapes.get(i));
        // // store.scores.add(store1.scores.get(i));
        // // }
        // // for (int i = 1; i < models.size(); i++) {
        // // ShapeletTree tree = models.get(i);
        // // ShapeletTree.ShapeStore store2 = tree.getStore();
        // // for (int j = 0; j < store2.scores.size(); j++) {
        // // store1.add(store2.shapes.get(j), store2.scores.get(j));
        // // }
        // // }
        // //
        // // QuickSort.quickSort(0, store1.scores.size(),
        // // (a, b) -> Double.compare(store1.scores.get(b), store1.scores.get(a)), (a, b) -> {
        // // Collections.swap(store1.scores, a, b);
        // // Collections.swap(store1.shapes, a, b);
        // // Collections.swap(store1.counts, a, b);
        // // Collections.swap(store1.normalizedShapes, a, b);
        // // });
        //
        // System.out.println(store1.scores);
        // System.out.println(store1.shapes.stream().map(Vector::size).collect(Collectors.toList()));
        // System.out.println(store1.counts);
        // System.out.println(store1.scores.size());
        lenSum.apply(v -> v / size());
        posSum.apply(v -> v / size());

        Map<Object, Integer> counts = Outputs.valueCounts(y); // TODO: 3/11/16 might be wrong
        DoubleArray apriori = DoubleArray.zeros(classes.size());
        for (int i = 0; i < classes.size(); i++) {
          apriori.set(i, counts.get(classes.get(i)) / (double) y.size());
        }

        return new RandomShapeletForest(classes, apriori, models, lenSum, posSum, oobIndicator);
      } catch (Exception e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }

    }

    @Override
    public String toString() {
      return "Ensemble of Randomized Shapelet Trees";
    }

    private static final class FitTask implements Callable<ShapeletTree> {

      private final ClassSet classSet;
      private final Input<? extends Vector> x;
      private final Output<?> y;
      private final List<?> classes;
      private final ShapeletTree.Configurator configurator;
      private final BooleanArray oobIndicator;


      private FitTask(ClassSet classSet, Input<? extends Vector> x, Output<?> y,
          ShapeletTree.Configurator configurator, List<?> classes, BooleanArray oobIndicator) {
        this.classSet = classSet;
        this.x = x;
        this.y = y;
        this.classes = classes;
        this.configurator = configurator;
        this.oobIndicator = oobIndicator;
      }

      @Override
      public ShapeletTree call() throws Exception {
        Random random = new Random(Thread.currentThread().getId() * System.nanoTime());
        ClassSet sample = sample(classSet, random);
        double low = configurator.lowerLength;
        double high = configurator.upperLength;
        return new ShapeletTree.Learner(low, high, configurator, sample, classes).fit(x, y);
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
