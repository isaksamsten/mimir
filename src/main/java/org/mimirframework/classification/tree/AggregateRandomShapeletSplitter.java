/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.classification.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataseries.Approximations;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.shapelet.IndexSortedNormalizedShapelet;
import org.mimirframework.shapelet.Shapelet;

/**
 * @author Isak Karlsson
 */
public class AggregateRandomShapeletSplitter extends RandomShapeletSplitter {

  private double aggregateFraction = 0.5;

  public AggregateRandomShapeletSplitter(Builder builder) {
    super(builder);
  }

  @Override
  public org.mimirframework.classification.tree.TreeSplit<ShapeletThreshold> find(org.mimirframework.classification.tree.ClassSet classSet, DataFrame x, Vector y) {
    int timeSeriesLength = x.columns();
    int upper = getUpperLength();
    int lower = getLowerLength();

    if (upper < 0) {
      upper = timeSeriesLength;
    }
    if (lower < 2) {
      lower = 2;
    }

    if (Math.addExact(upper, lower) > timeSeriesLength) {
      upper = timeSeriesLength - lower;
    }

    int size = (int) Math.round(timeSeriesLength * aggregateFraction);
    DataFrame paaX = Approximations.paa(x, size);
    List<Shapelet> shapelets = new ArrayList<>();
    for (int i = 0; i < getInspectedShapelets(); i++) {
      int index = classSet.getRandomSample().getRandomExample().getIndex();
      Vector t = paaX.loc().getRecord(index);
      int length = random.nextInt(upper) + lower;
      int start = random.nextInt(timeSeriesLength - length);
      shapelets.add(new DownsampledShapelet(index, aggregateFraction, start, length, t));
    }

    org.mimirframework.classification.tree.TreeSplit<ShapeletThreshold> threshold = super.findBestSplit(classSet, paaX, y, shapelets);
    DownsampledShapelet best = (DownsampledShapelet) threshold.getThreshold().getShapelet();

    IndexSortedNormalizedShapelet s =
        new IndexSortedNormalizedShapelet(best.start, best.length, x.loc().getRecord(best.index));

    return super.findBestSplit(classSet, x, y, Arrays.asList(s));
  }


  private static class DownsampledShapelet extends IndexSortedNormalizedShapelet {
    private final int start;
    private final int length;
    private final int index;

    public DownsampledShapelet(int index, double agg, int start, int length, Vector vector) {
      super((int) Math.round(start * agg), (int) Math.round(length * agg), vector);

      this.start = start;
      this.length = length;
      this.index = index;
    }
  }
}
