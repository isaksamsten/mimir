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
import org.briljantframework.data.Na;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.tree.pattern.PatternDistance;
import org.briljantframework.mimir.classification.tree.pattern.PatternTree;
import org.briljantframework.mimir.distance.Distance;
import org.briljantframework.mimir.distance.EarlyAbandonSlidingDistance;

/**
 * Created by isak on 3/17/16.
 */
public class ShapeletDistance implements PatternDistance<Vector, Shapelet> {
  private final Distance<Vector> categoricDistance = new ZeroOneDistance();
  private final Distance<Vector> numericDistance = new EarlyAbandonSlidingDistance();


  private static class ZeroOneDistance implements Distance<Vector> {

    @Override
    public double compute(Vector a, Vector b) {
      if (a instanceof Shapelet) {
        return b.loc().indexOf(a.loc().get(0)) < 0 ? 1 : 0;
      }
      return a.loc().indexOf(b.loc().get(0)) < 0 ? 1 : 0;
    }
  }


  @Override
  public double computeDistance(Vector record, Shapelet shapelet) {
    double distance;
    if (shapelet instanceof ChannelShapelet) {
      int channelIndex = ((ChannelShapelet) shapelet).getChannel();
      Vector channel = record.loc().get(Vector.class, channelIndex);
      if (shapelet.getDelegate() instanceof CategoricShapelet) {
        if (Is.NA(channel)) {
          distance = Na.DOUBLE;
        } else {
          distance = categoricDistance.compute(channel, shapelet);
        }
      } else {
        if (Is.NA(channel)) {
          distance = Na.DOUBLE;
        } else {
          distance = numericDistance.compute(channel, shapelet);
        }
      }
    } else {
      distance = numericDistance.compute(record, shapelet);
    }
    return distance;
  }
}
