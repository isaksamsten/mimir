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
package org.briljantframework.mimir.shapelet;

import org.briljantframework.data.Is;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.tree.pattern.SamplingPatternFactory;
import org.briljantframework.primitive.IntList;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by isak on 3/17/16.
 */
public class ShapeletFactory extends SamplingPatternFactory<Vector, Shapelet> {
  private final double lowerLength = 0.025, upperLength = 1;

  private static IntList nonNaIndicies(Vector vector) {
    IntList nonNas = new IntList();
    for (int i = 0; i < vector.size(); i++) {
      if (!Is.NA(vector.loc().get(i))) {
        nonNas.add(i);
      }
    }
    return nonNas;
  }

  private static boolean isCategorical(Vector timeSeries) {
    return timeSeries != null && String.class.isAssignableFrom(timeSeries.getType().getDataClass());
  }

  private Shapelet getUnivariateShapelet(Vector timeSeries) {
    if (timeSeries == null) {
      return null;
    }
    if (isCategorical(timeSeries)) {
      int rnd = ThreadLocalRandom.current().nextInt(timeSeries.size());
      return new CategoricShapelet(timeSeries.loc().get(String.class, rnd));
    }

    int timeSeriesLength = timeSeries.size();
    int upper = (int) Math.round(timeSeriesLength * upperLength);
    int lower = (int) Math.round(timeSeriesLength * lowerLength);
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
      // return new Shapelet(0, 1, timeSeries);
      return null;
    }

    int length = ThreadLocalRandom.current().nextInt(upper) + lower;
    int start = ThreadLocalRandom.current().nextInt(timeSeriesLength - length);
    Shapelet shapelet;
    if (isCategorical(timeSeries)) {
      shapelet = new CategoricShapelet(start, length, timeSeries);
    } else {
      // TODO: normalization should be a param
      // TODO: normalized
      shapelet = new IndexSortedNormalizedShapelet(start, length, timeSeries);
    }
    return shapelet;
  }

  @Override
  public Shapelet createPattern(Vector timeSeries) {
    Shapelet shapelet;
    // MTS
    if (Vector.class.isAssignableFrom(timeSeries.getType().getDataClass())) {
      IntList nonNas = nonNaIndicies(timeSeries);
      if (!nonNas.isEmpty()) {
        int channelIndex = nonNas.get(ThreadLocalRandom.current().nextInt(nonNas.size()));
        Vector channel = timeSeries.loc().get(Vector.class, channelIndex);
        Shapelet univariateShapelet = getUnivariateShapelet(channel);
        if (univariateShapelet == null) {
          shapelet = null;
        } else {
          shapelet = new ChannelShapelet(channelIndex, univariateShapelet);
        }
      } else {
        shapelet = null;
      }
    } else {
      shapelet = getUnivariateShapelet(timeSeries);
    }
    return shapelet;
  }
}
