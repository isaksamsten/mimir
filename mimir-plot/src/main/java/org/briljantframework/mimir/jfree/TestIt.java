package org.briljantframework.mimir.jfree;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.briljantframework.array.Array;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
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
    return Plots.bar(Vector.of(10, 20, 30));
  }

  public static Plot testStatisticalDataFrameChart() {
    RealDistribution distribution = new UniformRealDistribution(0, 200);
    DataFrame df =
        DataFrame.of("A", Vector.fromSupplier(distribution::sample, 100), "B",
            Vector.fromSupplier(distribution::sample, 100));

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
    XYDataset ds = new ArrayXYDataset(x, y, Array.of("A", "B").toList());

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
