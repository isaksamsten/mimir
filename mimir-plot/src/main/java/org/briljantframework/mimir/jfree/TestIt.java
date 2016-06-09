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
package org.briljantframework.mimir.jfree;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
class TestIt {

  public static Plot testArrayYDataset() {
    return Plots.line(Arrays.randn(100).reshape(10, 10));
  }

  public static Plot testVectorBarChart() {
    return Plots.bar(Series.of(10, 20, 30));
  }

  public static Plot testStatisticalDataFrameChart() {
    RealDistribution distribution = new UniformRealDistribution(0, 200);
    DataFrame df =
        DataFrame.of("A", Series.generate(distribution::sample, 100), "B",
            Series.generate(distribution::sample, 100));

    return Plots.statisticalBar(df);
  }

  public static XYPlot testErrorLine() {
    DoubleArray y = DoubleArray.linspace(-4, 4, 100).reshape(50, 2);
    DoubleArray x = DoubleArray.linspace(-4, 4, 100).reshape(50, 2);
    DoubleArray error = Arrays.randn(100).reshape(50, 2);
    return Plots.line(x.boxed(), y.boxed(), error.boxed());
  }

  public static void main(String[] args) {
    // JFreeChart chart = testVectorBarChart();
    // JFreeChart chart = testStatisticalDataFrameChart();
    Array<Integer> x = Array.of(1, 2, 3, 4, 5, 6).reshape(3, 2);
    Array<Integer> y = Array.of(1, 2, 3, 4, 5, 6).reshape(3, 2);
    XYDataset ds = new ArrayXYDataset(x, y, java.util.Arrays.asList("A", "B"));

    XYPlot chart =
        new XYPlot(ds, new NumberAxis(), new NumberAxis(), new XYLineAndShapeRenderer(true, true));
    // Plots.line(DoubleArray.of(1, 2, 3, 4, 5, 6).reshape(3, 2),
    // DoubleArray.of(1, 2, 3, 4, 5, 6).reshape(3, 2));

    // chart.setDataset(
    // 1,
    // new ArrayXYDataset(Arrays.randn(100).reshape(100, 1).boxed(), Arrays.randn(100)
    // .reshape(100, 1).boxed()));
    // chart.setRenderer(1, new XYLineAndShapeRenderer(false, true));
    Plots.show(chart);
  }
}
