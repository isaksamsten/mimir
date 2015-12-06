package org.briljantframework.mimir.jfree;

import java.awt.*;
import java.util.AbstractList;
import java.util.Iterator;
import java.util.List;

import javax.swing.*;

import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartTheme;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.DefaultDrawingSupplier;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.renderer.category.StatisticalBarRenderer;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.renderer.xy.XYErrorRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.statistics.StatisticalCategoryDataset;
import org.jfree.data.xy.XYDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class Plots {

  private static final Paint[] COLORS = new Paint[] {new Color(0, 55, 255, 180),
      new Color(255, 172, 0, 180), new Color(128, 0, 255, 180), new Color(0, 205, 0, 180),
      new Color(205, 0, 0, 180), new Color(255, 215, 0, 180), new Color(255, 0, 255, 180),
      new Color(255, 166, 201, 180), new Color(207, 207, 207, 180), new Color(0, 255, 255, 180),
      new Color(102, 56, 10, 180), new Color(0, 0, 0, 180)};

  private static final ChartTheme DEFAULT_THEME = new MimirChartTheme("Mimir");

  /**
   * Provides an infinite (circular) list of colors. For example,
   * {@code DEFAULT_COLORS.stream().take(10)} to get the first 10 collections.
   */
  public static final List<Paint> DEFAULT_COLORS = new AbstractList<Paint>() {
    @Override
    public Paint get(int index) {
      return COLORS[index % COLORS.length];
    }

    @Override
    public int size() {
      return COLORS.length;
    }
  };

  public static Iterator<Paint> colorIterator() {
    return DEFAULT_COLORS.iterator();
  }

  private static class MimirChartTheme extends StandardChartTheme {
    public MimirChartTheme(String name) {
      this(name, false);
    }

    public MimirChartTheme(String name, boolean shadow) {
      super(name, shadow);
      setBarPainter(new StandardBarPainter());
      setXYBarPainter(new StandardXYBarPainter());

      setChartBackgroundPaint(Color.white);
      setPlotBackgroundPaint(Color.white);
      setPlotOutlinePaint(Color.black);
      setDomainGridlinePaint(Color.darkGray);
      setRangeGridlinePaint(Color.darkGray);

      String font = "Sans";
      setLargeFont(new Font(font, Font.PLAIN, 11));
      setExtraLargeFont(new Font(font, Font.BOLD, 12));
      setRegularFont(new Font(font, Font.PLAIN, 9));
      setSmallFont(new Font(font, Font.PLAIN, 7));
      setShadowVisible(false);
      setDrawingSupplier(new DefaultDrawingSupplier(COLORS, COLORS, new Stroke[] {new BasicStroke(
          1.0f)}, new Stroke[] {new BasicStroke(0.5f)},
          DefaultDrawingSupplier.DEFAULT_SHAPE_SEQUENCE));
    }
  }

  private static XYPlot postProcessPlot(XYPlot plot) {
    plot.setRangeGridlinesVisible(false);
    plot.setDomainGridlinesVisible(false);
    return plot;
  }

  private static CategoryPlot postProcessPlot(CategoryPlot plot) {
    plot.setRangeGridlinesVisible(false);
    plot.setDomainGridlinesVisible(false);
    return plot;
  }

  private static JFreeChart postProcessChart(JFreeChart chart) {
    DEFAULT_THEME.apply(chart);
    return chart;
  }

  public static JFreeChart applyTheme(JFreeChart chart) {
    return postProcessChart(chart);
  }

  private static XYPlot createLinePlot(XYDataset dataset) {
    XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(true, false);
    XYPlot plot = new XYPlot(dataset, new NumberAxis(), new NumberAxis(), renderer);
    return postProcessPlot(plot);
  }

  public static XYPlot line(Array<? extends Number> y) {
    return createLinePlot(new ArrayYDataset(y));
  }

  public static XYPlot line(DoubleArray y) {
    return line(y.boxed());
  }

  public static XYPlot line(Array<? extends Number> x, Array<? extends Number> y) {
    return createLinePlot(new ArrayXYDataset(x, y));
  }

  public static XYPlot line(DoubleArray x, DoubleArray y) {
    return line(x.boxed(), y.boxed());
  }

  private static XYPlot createScatterPlot(XYDataset dataset) {
    XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(false, true);
    XYPlot plot = new XYPlot(dataset, new NumberAxis(), new NumberAxis(), renderer);
    return postProcessPlot(plot);
  }

  public static XYPlot scatter(Array<? extends Number> x, Array<? extends Number> y) {
    XYPlot plot = createScatterPlot(new ArrayXYDataset(x, y));
    return plot;
  }

  public static XYPlot scatter(DoubleArray x, DoubleArray y) {
    return scatter(x.boxed(), y.boxed());
  }

  public static XYPlot line(Array<? extends Number> x, Array<? extends Number> y,
      Array<? extends Number> error) {
    XYPlot plot = createLinePlot(new ArrayIntervalXYDataset(x, y, error));
    plot.setRenderer(new XYErrorRenderer());
    return plot;
  }

  private static CategoryPlot createBarPlot(CategoryDataset dataset) {
    CategoryItemRenderer renderer = new BarRenderer();
    CategoryPlot plot = new CategoryPlot(dataset, new CategoryAxis(), new NumberAxis(), renderer);
    return postProcessPlot(plot);
  }

  public static CategoryPlot bar(Vector data) {
    return createBarPlot(new VectorCategoryDataset(data));
  }

  private static CategoryPlot createStatistaicalBarPlot(StatisticalCategoryDataset dataset) {
    CategoryItemRenderer renderer = new StatisticalBarRenderer();
    CategoryPlot plot = new CategoryPlot(dataset, new CategoryAxis(), new NumberAxis(), renderer);
    return postProcessPlot(plot);
  }

  public static CategoryPlot statisticalBar(DataFrame df) {
    return createStatistaicalBarPlot(new StatisticalDataFrameDataset(df));
  }

  public static void show(Plot plot) {
    JFreeChart chart = postProcessChart(new JFreeChart(plot));
    chart.setAntiAlias(true);
    ChartFrame frame = new ChartFrame("Chart window", chart);
    frame.pack();
    frame.setVisible(true);
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
  }

}
