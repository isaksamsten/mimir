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

import java.util.*;

import org.briljantframework.DoubleVector;
import org.briljantframework.mimir.distance.Distance;

/**
 * Created by isak on 2017-03-28.
 */
public class ShapeletStorage {

  private final Map<Integer, List<ShapeletScore>> shapelets = new HashMap<>();
  private final Distance<DoubleVector> distance;
  private final double lambda;

  public ShapeletStorage(Distance<DoubleVector> distance, double lambda) {
    this.distance = distance;
    this.lambda = lambda;
  }

  public void add(Shapelet shapelet, double score) {
    List<ShapeletScore> scores = shapelets.get(shapelet.size());
    if (scores == null) {
      scores = new ArrayList<>();
      scores.add(new ShapeletScore(shapelet, score));
    } else {
      boolean increment = false;
      for (ShapeletScore shapeletScore : scores) {
        double dist = distance.compute(shapeletScore.getPrototypeShapelet(), shapelet);
        if (dist <= lambda) {
          shapeletScore.incrementScore(shapelet, score);
          increment = true;
        }
      }
      if (!increment) {
        scores.add(new ShapeletScore(shapelet, score));
      }
    }
    shapelets.put(shapelet.size(), scores);
  }

  public List<ShapeletScore> get(int size) {
    return shapelets.get(size);
  }

  public List<ShapeletScore> getSorted() {
    List<ShapeletScore> scores = new ArrayList<>();
    for (List<ShapeletScore> shapeletScores : shapelets.values()) {
      scores.addAll(shapeletScores);
    }

    scores.sort(Comparator.reverseOrder());
    return scores;
  }

}
