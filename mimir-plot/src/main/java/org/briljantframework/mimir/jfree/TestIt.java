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

import static org.briljantframework.mimir.data.transform.Transformations.meanImputer;
import static org.briljantframework.mimir.data.transform.Transformations.zNormalization;
import static org.briljantframework.mimir.data.transform.Transformations.pairwiseDistance;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;
import org.briljantframework.mimir.data.Instance;
import org.briljantframework.mimir.data.transform.Transformation;
import org.briljantframework.mimir.distance.EuclideanDistance;
import org.briljantframework.mimir.mds.MultidimensionalScaling;
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
    DataFrame df = DataFrame.of("A", Series.generate(distribution::sample, 100), "B",
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
    // Plot chart = testStatisticalDataFrameChart();
    // Array<Integer> x = Array.of(1, 2, 3, 4, 5, 6).reshape(3, 2);
    // Array<Integer> y = Array.of(1, 2, 3, 4, 5, 6).reshape(3, 2);
    //
    // ArrayDataSource data = new ArrayDataSource(x);
    //
    // de.erichseifert.gral.plots.XYPlot plot = new de.erichseifert.gral.plots.XYPlot(data);
    // JFrame frame = new JFrame();
    // frame.add(new InteractivePanel(plot));
    // frame.setVisible(true);
    //
    // InputTransformation<Instance, Instance> transformation = new Normalizer<>();
    // InputTransformer<Instance, Instance> transformer = transformation.fit(in);
    //
    //
    // System.out.println(transformer.transform(in));

    // Transformation<String, Double> f = s -> (f1) -> (double) s.length();
    // Transformation<Double, Integer> d = s -> (f1) -> (int) Math.floor(s * f1);
    //
    //
    // Transformation<String, Integer> x = f.followedBy(d);
    // Transformer<String, Integer> hello = x.fit("hello");
    // System.out.println(hello.transform("Isak"));

    DataFrame x = Datasets.loadIris();
    Input<Instance> in = Inputs.asInput(x.drop("Class"));
    Transformation<Input<Instance>, DoubleArray> f = meanImputer().followedBy(zNormalization())
        .followedBy(pairwiseDistance(EuclideanDistance.getInstance()))
        .followedBy(new MultidimensionalScaling(2));

    DoubleArray mds = f.fitTransform(in);

    XYDataset cls1 = new ArrayXYDataset(mds.get(Range.of(50), IntArray.of(0)).boxed(),
        mds.get(Range.of(0, 50), IntArray.of(1)).boxed());
    XYDataset cls2 = new ArrayXYDataset(mds.get(Range.of(50, 100), IntArray.of(0)).boxed(),
        mds.get(Range.of(50, 100), IntArray.of(1)).boxed());
    XYDataset cls3 = new ArrayXYDataset(mds.get(Range.of(100, 150), IntArray.of(0)).boxed(),
        mds.get(Range.of(100, 150), IntArray.of(1)).boxed());



    XYLineAndShapeRenderer cls1Renderer = new XYLineAndShapeRenderer(false, true);
    XYPlot scatter = new XYPlot(cls1, new NumberAxis("component_0"), new NumberAxis("component_1"), cls1Renderer);

    scatter.setDataset(1, cls2);

    XYLineAndShapeRenderer cls2Renderer = new XYLineAndShapeRenderer(false, true);
    cls2Renderer.setBaseFillPaint(Plots.DEFAULT_COLORS.get(1));
    scatter.setRenderer(1, cls2Renderer);

    XYLineAndShapeRenderer cls3Renderer = new XYLineAndShapeRenderer(false, true);
    cls2Renderer.setBaseFillPaint(Plots.DEFAULT_COLORS.get(2));
    scatter.setRenderer(2, cls3Renderer);
    scatter.setDataset(2, cls3);



    Plots.show(scatter);


    // System.out.println(x);
    // System.out.println(y);
    // XYDataset ds = new ArrayXYDataset(x, y, java.util.Arrays.asList("A", "B"));
    ////
    // XYPlot chart =
    // new XYPlot(ds, new NumberAxis(), new NumberAxis(), new XYLineAndShapeRenderer(true, true));
    // Plots.line(DoubleArray.of(1, 2, 3, 4, 5, 6).reshape(3, 2),
    // DoubleArray.of(1, 2, 3, 4, 5, 6).reshape(3, 2));
    //
    // chart.setDataset(
    // 1,
    // new ArrayXYDataset(Arrays.randn(100).reshape(100, 1).boxed(), Arrays.randn(100)
    // .reshape(100, 1).boxed()));
    // chart.setRenderer(1, new XYLineAndShapeRenderer(false, true));
    // Plots.show(chart);
  }
}
