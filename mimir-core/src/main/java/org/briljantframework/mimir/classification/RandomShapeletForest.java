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

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.Property;
import org.briljantframework.mimir.classification.tree.ClassSet;
import org.briljantframework.mimir.classification.tree.pattern.PatternDistance;
import org.briljantframework.mimir.classification.tree.pattern.PatternFactory;
import org.briljantframework.mimir.classification.tree.pattern.RandomPatternForest;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.distance.EarlyAbandonSlidingDistance;
import org.briljantframework.mimir.shapelet.IndexSortedNormalizedShapelet;
import org.briljantframework.mimir.shapelet.MultivariateShapelet;
import org.briljantframework.mimir.supervised.Predictor;
import org.briljantframework.mimir.timeseries.data.MultivariateTimeSeries;
import org.briljantframework.mimir.timeseries.data.MultivariateTimeSeriesSchema;
import org.briljantframework.mimir.timeseries.data.TimeSeries;

/**
 * Created by isak on 2017-01-26.
 */
public class RandomShapeletForest<Out> implements Classifier<MultivariateTimeSeries, Out>,
    ProbabilityEstimator<MultivariateTimeSeries, Out> {

  public static Property<Double> LOWER =
      Property.of("lower", Double.class, 0.025, i -> i > 0 && i < 1);
  public static Property<Double> UPPER =
      Property.of("upper", Double.class, 1.0, i -> i > 0 && i < 1);

  private final RandomPatternForest<MultivariateTimeSeries, Out> forest;
  private final MultivariateTimeSeriesSchema schema;

  private RandomShapeletForest(MultivariateTimeSeriesSchema schema,
      RandomPatternForest<MultivariateTimeSeries, Out> forest) {
    this.schema = schema;
    this.forest = forest;
  }

  public List<ProbabilityEstimator<MultivariateTimeSeries, Out>> getMembers() {
    return forest.getEnsembleMembers();
  }

  @Override
  public List<Out> predict(Input<MultivariateTimeSeries> x) {
    Check.argument(schema.equals(x.getSchema()), "illegal input schema");
    return forest.predict(x);
  }

  @Override
  public Array<Out> getClasses() {
    return forest.getClasses();
  }

  @Override
  public DoubleArray estimate(MultivariateTimeSeries input) {
    Check.argument(schema.isValid(input), "illegal input");
    return forest.estimate(input);
  }

  public static final class Learner<Out>
      extends Predictor.Learner<MultivariateTimeSeries, Out, RandomShapeletForest<Out>> {

    private final static PatternDistance<MultivariateTimeSeries, MultivariateShapelet> PATTERN_DISTANCE =
        new PatternDistance<MultivariateTimeSeries, MultivariateShapelet>() {
          private EarlyAbandonSlidingDistance distance = new EarlyAbandonSlidingDistance();

          public double computeDistance(MultivariateTimeSeries a, MultivariateShapelet b) {
            return distance.compute(a.getDimension(b.getDimension()), b.getShapelet());
          }
        };

    @Override
    public RandomShapeletForest<Out> fit(Input<MultivariateTimeSeries> in, List<Out> out) {
      Check.argument(in.getSchema() instanceof MultivariateTimeSeriesSchema);
      RandomPatternForest.Learner<MultivariateTimeSeries, Out> forest =
          new RandomPatternForest.Learner<>(
              getPatternFactory(getOrDefault(LOWER), getOrDefault(UPPER)), PATTERN_DISTANCE,
              getOrDefault(Ensemble.SIZE));
      return new RandomShapeletForest<>((MultivariateTimeSeriesSchema) in.getSchema(),
          forest.fit(in, out));
    }

    private static PatternFactory<MultivariateTimeSeries, MultivariateShapelet> getPatternFactory(
        final double lowFrac, final double uppFrac) {
      return new PatternFactory<MultivariateTimeSeries, MultivariateShapelet>() {

        /**
         * @param inputs the input dataset
         * @param classSet the inputs included in the current bootstrap.
         * @return a shapelet
         */
        public MultivariateShapelet createPattern(Input<? extends MultivariateTimeSeries> inputs,
            ClassSet classSet) {
          MultivariateTimeSeries mts =
              inputs.get(classSet.getRandomSample().getRandomExample().getIndex());
          ThreadLocalRandom random = ThreadLocalRandom.current();
          int randomDim = random.nextInt(mts.dimensions());
          TimeSeries uts = mts.getDimension(randomDim);
          int timeSeriesLength = uts.size();
          int lower = (int) Math.round(timeSeriesLength * lowFrac);
          int upper = (int) Math.round(timeSeriesLength * uppFrac);
          if (lower < 2) {
            lower = 2;
          }

          if (Math.addExact(upper, lower) > timeSeriesLength) {
            upper = timeSeriesLength - lower;
          }
          if (lower == upper) {
            upper -= 2;
          }
          if (upper < 1) {
            return null;
          }

          int length = ThreadLocalRandom.current().nextInt(upper) + lower;
          int start = ThreadLocalRandom.current().nextInt(timeSeriesLength - length);
          return new MultivariateShapelet(randomDim,
              new IndexSortedNormalizedShapelet(start, length, uts));
        }
      };
    }


  }

}
