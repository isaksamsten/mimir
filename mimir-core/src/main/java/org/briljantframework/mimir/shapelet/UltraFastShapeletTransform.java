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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.briljantframework.data.Na;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.MixedDataFrame;
import org.briljantframework.data.dataframe.transform.Transformation;
import org.briljantframework.data.dataframe.transform.Transformer;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.VectorType;
import org.briljantframework.mimir.distance.Distance;
import org.briljantframework.mimir.distance.EarlyAbandonSlidingDistance;
import org.briljantframework.mimir.distance.EuclideanDistance;

/**
 * [1] M.Wistuba at al. Ultra-Fast Shapelets for Time Series Classication.
 * http://arxiv.org/pdf/1503.05018.pdf
 */
public class UltraFastShapeletTransform implements Transformation {

  private final int shapelets;
  private final double p;
  private final double upperLength = 1;
  private final double lowerLength = 0.025;
  private final Distance<Vector> numericDistance =
      new EarlyAbandonSlidingDistance(EuclideanDistance.getInstance());

  public UltraFastShapeletTransform(int shapelets, double p) {
    // Check.argument(shapelets > 0);
    this.shapelets = shapelets;
    this.p = p;
  }

  @Override
  public Transformer fit(DataFrame df) {
    List<Shapelet> features = new ArrayList<>();
    boolean selectLowest =
        Vector.class.isAssignableFrom(df.loc().getRecord(0).getType().getDataClass());
    if (shapelets == -1) {
      int n = df.rows();
      int m = df.columns();
      int s = 1;

      if (selectLowest) {
        // m = df.loc().getRecord(0).loc().get(Vector.class, 0).size();
        m = df.getRecords().stream().mapToInt(r -> r.loc().get(Vector.class, 0).size()).max()
            .orElse(0);
        s = df.loc().getRecord(0).size();
      }

      double sum = 0;
      for (int i = 3; i <= m; i++) {
        sum += m * s - i + 1;
      }
      double f = p * (1 / sum);
      for (int i = 3; i <= m; i++) {
        long r = Math.round(f * (m * s - i + 1));
        for (int k = 0; k < s; k++) {
          for (int j = 0; j < r; j++) {
            int vec = ThreadLocalRandom.current().nextInt(n);
            Vector record = df.loc().getRecord(vec).get(Vector.class, k);
            if (record.size() + 1 - i <= 0) {
              continue;
            }
            int start = ThreadLocalRandom.current().nextInt(record.size() + 1 - i);
            features.add(new IndexSortedNormalizedShapelet(start, i, record));
          }
        }
      }
      System.out.println(features.size());
    } else {
      for (int i = 0; i < shapelets; i++) {
        int rIndex = ThreadLocalRandom.current().nextInt(df.rows());
        Vector record = df.loc().getRecord(rIndex);
        Shapelet feature;
        // MTS
        if (Vector.class.isAssignableFrom(record.getType().getDataClass())) {
          int cIndex = ThreadLocalRandom.current().nextInt(record.size()); // size() == no-channels
          Shapelet shapelet = sample(record.loc().get(Vector.class, cIndex));
          if (shapelet == null) {
            i--;
            continue;
          } else {
            feature = new ChannelShapelet(cIndex, shapelet);
          }
        } else {
          feature = sample(record);
        }
        if (feature == null) {
          i--;
        } else {
          features.add(feature);
        }
      }
      // selectLowest = false;
    }

    return new ShapeletTransformer(numericDistance, features, selectLowest);
  }


  private Shapelet sample(Vector timeSeries) {
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
    return new IndexSortedNormalizedShapelet(start, length, timeSeries);
  }

  private static class ShapeletTransformer implements Transformer {
    private final List<Shapelet> features;
    private final Distance<Vector> numericDistance;
    private final boolean selectLowest;

    public ShapeletTransformer(Distance<Vector> numericDistance, List<Shapelet> features,
        boolean selectLowest) {
      this.features = features;
      this.numericDistance = numericDistance;
      this.selectLowest = selectLowest;
    }

    @Override
    public DataFrame transform(DataFrame x) {
      DataFrame.Builder out = new MixedDataFrame.Builder();
      for (int i = 0; i < features.size(); i++) {
        out.set("Shapelet: " + i, VectorType.DOUBLE);
      }

      // ensures that the vectors are not resized
      for (int i = 0; i < features.size(); i++) {
        out.loc().set(x.rows() - 1, i, Na.DOUBLE);
      }

      if (selectLowest) {
        IntStream.range(0, x.rows()).parallel().forEach(i -> {
          // for (int i = 0; i < x.rows(); i++) {
          Vector record = x.loc().getRecord(i);
          for (int j = 0; j < features.size(); j++) {
            // System.err.println("doin " + i + " : " + j);
            Shapelet shapelet = features.get(j);
            double minDist = Double.POSITIVE_INFINITY;
            for (int k = 0; k < record.size(); k++) {
              Vector channel = record.loc().get(Vector.class, k);
              double dist = numericDistance.compute(channel, shapelet);
              if (dist < minDist) {
                minDist = dist;
              }
              if (minDist == 0) {
                break;
              }
            }
            synchronized (out) {
              out.loc().set(i, j, minDist);
            }
            // }
          }
        });
      } else {
        for (int i = 0; i < x.rows(); i++) {
          Vector record = x.loc().getRecord(i);
          for (int j = 0; j < features.size(); j++) {
            Shapelet shapelet = features.get(j);
            Vector channel = record;
            if (shapelet instanceof ChannelShapelet) {
              channel = record.loc().get(Vector.class, ((ChannelShapelet) shapelet).getChannel());
            }
            out.loc().set(i, j, numericDistance.compute(channel, shapelet));
          }
        }
      }
      return out.build();
    }
  }
}
