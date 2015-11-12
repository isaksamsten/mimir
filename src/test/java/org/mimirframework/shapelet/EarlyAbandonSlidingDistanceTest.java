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

package org.mimirframework.shapelet;

import org.junit.Test;

public class EarlyAbandonSlidingDistanceTest {

  @Test
  // TODO - bug!
  public void testDistance() throws Exception {
    // DoubleMatrix a = DefaultDoubleMatrix.of(1, 6, 1, 2, 3, 4, 5, 6);
    // DoubleMatrix a1 = DefaultDoubleMatrix.of(1, 4, 2, 3, 4, 6);
    // DoubleMatrix a2 = DefaultDoubleMatrix.of(1, 6, 0, 1, 2, 4, 2, 3);
    // DoubleMatrix b = DefaultDoubleMatrix.of(1, 2, 2, 3);
    //
    // Shapelet shapelet = new IndexSortedNormalizedShapelet(1, 2, Convert.toAdapter(a));
    // System.out.println(shapelet);
    // System.out.println(b);

    // System.out.println(shapelet);
    // System.out.println(new IndexSortedNormalizedShapelet(0, a2.size(), Convert.toAdapter(a2)));

    // Distance distance = EarlyAbandonSlidingDistance.create(Euclidean.getInstance());
    // System.out.println(distance.distance(a, shapelet));
    // System.out.println(distance.distance(a1, shapelet));
    // System.out.println(distance.compute(Convert.toAdapter(a2), shapelet));

    // distance = OnlineReorderEarlyAbandonSlidingDistance.create();

    // System.out.println(distance.compute(Convert.toAdapter(a2), shapelet));

    // System.out.println(Distance.EUCLIDEAN.distance(b, b));
  }

  @Test
  public void testLong() throws Exception {
    // Matrix l = DenseMatrix.of(1, 22, 3, 3, 2, 1, 4, 32, 5, 6, 7, 7, 8, 8, 1, 2, 1, 4, 32, 432, 1,
    // 34, 5, 32);
    // Shapelet candidate = NormalizedShapelet.create(5, 10, l);
    //
    //
    // Vector l = Convert.toAdapter(Matrices.(1, 3213));
    // Shapelet candidate = new Shapelet(2314, 500, l);

    // Matrix l = DenseMatrix.of(1, 6, 1, 2, 3, 4, 5543, 6);
    // Shapelet candidate = NormalizedShapelet.create(2, 4, l);

    // Distance distance = EarlyAbandonSlidingDistance.create(Euclidean.getInstance());
    //
    // long s = System.currentTimeMillis();
    // for (int i = 0; i < 100; i++) {
    // blacbox(distance.distance(l, candidate));
    // }
    // System.out.println(distance.compute(l, candidate));
    // System.out.println(System.currentTimeMillis() - s);
    //
    // Distance d = OnlineReorderEarlyAbandonSlidingDistance.create();
    // s = System.currentTimeMillis();
    // for (int i = 0; i < 100; i++) {
    // blacbox(d.distance(l, candidate));
    // }

    // System.out.println(d.compute(l, candidate));
    // System.out.println(System.currentTimeMillis() - s);
    //
  }

  private void blacbox(double distance) {

  }

  @Test
  public void testName() throws Exception {
    // DataSeriesInputStream tsis = new DataSeriesInputStream(new
    // FileInputStream("erlang/synthetic_control_TRAIN
    // .txt"));
    // Container<Frame, DefaultTarget> container = tsis.read(Frame.FACTORY, DefaultTarget.FACTORY);
    //
    // VectorView a = container.getDataset().getEntry(69);
    // Shapelet shapelet = Shapelet.create(3, 50, container.getDataset().getEntry(70));
    //
    // System.out.println(a.copy());
    // System.out.println(container.getDataset().getRowView(70));
    // System.out.println(shapelet);
    //
    // Distance distance = EarlyAbandonSlidingDistance.create(Distance.EUCLIDEAN);
    // System.out.println(1 / (double) shapelet.size() * distance.distance(a, shapelet));
  }
}
