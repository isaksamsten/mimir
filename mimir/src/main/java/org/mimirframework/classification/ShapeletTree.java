package org.mimirframework.classification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataseries.Aggregator;
import org.briljantframework.data.dataseries.Approximations;
import org.briljantframework.data.dataseries.MeanAggregator;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.briljantframework.statistics.FastStatistics;
import org.mimirframework.classification.tree.ClassSet;
import org.mimirframework.classification.tree.Example;
import org.mimirframework.classification.tree.Gain;
import org.mimirframework.classification.tree.Impurity;
import org.mimirframework.classification.tree.ShapeletThreshold;
import org.mimirframework.classification.tree.TreeBranch;
import org.mimirframework.classification.tree.TreeClassifier;
import org.mimirframework.classification.tree.TreeLeaf;
import org.mimirframework.classification.tree.TreeNode;
import org.mimirframework.classification.tree.TreeSplit;
import org.mimirframework.classification.tree.TreeVisitor;
import org.mimirframework.distance.Distance;
import org.mimirframework.distance.DynamicTimeWarping;
import org.mimirframework.distance.EarlyAbandonSlidingDistance;
import org.mimirframework.distance.EuclideanDistance;
import org.mimirframework.shapelet.ChannelShapelet;
import org.mimirframework.shapelet.DerivativeShapelet;
import org.mimirframework.shapelet.IndexSortedNormalizedShapelet;
import org.mimirframework.shapelet.Shapelet;
import org.mimirframework.supervised.Predictor;

import com.carrotsearch.hppc.IntDoubleMap;
import com.carrotsearch.hppc.IntDoubleOpenHashMap;
import com.carrotsearch.hppc.ObjectDoubleMap;
import com.carrotsearch.hppc.ObjectDoubleOpenHashMap;
import com.carrotsearch.hppc.ObjectIntMap;
import com.carrotsearch.hppc.ObjectIntOpenHashMap;
import com.carrotsearch.hppc.cursors.ObjectDoubleCursor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ShapeletTree extends TreeClassifier<ShapeletThreshold> {

  public ShapeStore getStore() {
    return store;
  }

  public static class ShapeStore {

    public static final double MIN_DIST = 5;
    public List<Shapelet> shapes = new ArrayList<>();
    public List<Double> scores = new ArrayList<>();
    public List<Integer> counts = new ArrayList<>();
    public List<DoubleArray> normalizedShapes = new ArrayList<>();

    private DynamicTimeWarping distance = new DynamicTimeWarping(-1);

    public void add(Shapelet shapelet, double score) {
//      if (shapes.isEmpty()) {
//        shapes.add(shapelet);
//        scores.add(score);
//        counts.add(1);
//        normalizedShapes.add(shapelet.toDoubleArray());
//        return;
//      }
//      int min = -1;
//      for (int i = 0; i < shapes.size(); i++) {
//        if (shapes.get(i).size() == shapelet.size()) {
//          min = i;
//          break;
//        }
//      }
//
//      if (min < 0) {
//        shapes.add(shapelet);
//        scores.add(score);
//        counts.add(1);
//        normalizedShapes.add(shapelet.toDoubleArray().div(score));
//      } else {
//        scores.set(min, scores.get(min) + score);
//        counts.set(min, counts.get(min) + 1);
//
//        DoubleArray arr = normalizedShapes.get(min);
//        for (int i = 0; i < arr.size(); i++) {
//          arr.set(i, arr.get(i) + shapelet.loc().getAsDouble(i) / score);
//        }
//      }//
       // int min = -1;
       // double minDist = Double.POSITIVE_INFINITY;
       // for (int i = 0; i < shapes.size(); i++) {
       // double dist = distance.compute(shapelet, shapes.get(i));
       // if (dist < minDist && dist < MIN_DIST) {
       // min = i;
       // minDist = dist;
       // }
       // }
       // if (min < 0) {
       // shapes.add(shapelet);
       // scores.add(score);
       // counts.add(1);
       // } else {
       // scores.set(min, scores.get(min) + score);
       // counts.set(min, counts.get(min) + 1);
       // }
    }
  }

  public final ClassSet classSet;
  private final int depth;
  private final DoubleArray lengthImportance;
  private final DoubleArray positionImportance;
  private final ShapeStore store;

  private ShapeletTree(Vector classes, TreeNode<ShapeletThreshold> node,
      TreeVisitor<ShapeletThreshold> predictionVisitor, DoubleArray lengthImportance,
      DoubleArray positionImportance, int depth, ClassSet classSet, ShapeStore store) {
    super(classes, node, predictionVisitor);
    this.lengthImportance = lengthImportance;
    this.positionImportance = positionImportance;
    this.depth = depth;
    this.classSet = classSet;
    this.store = store;
  }

  /**
   * Gets position importance.
   *
   * @return the position importance
   */
  public DoubleArray getPositionImportance() {
    return positionImportance;
  }

  /**
   * Gets length importance.
   *
   * @return the length importance
   */
  public DoubleArray getLengthImportance() {
    return lengthImportance;
  }

  public int getDepth() {
    return depth;
  }

  /**
   * An implementation of a shapelet tree
   *
   * <p>
   * <b>The code herein is so ugly that a kitten dies every time someone look at it.</b>
   *
   * @author Isak Karlsson
   */
  public static class Learner implements Predictor.Learner<ShapeletTree> {

    protected final Random random = new Random();
    protected final Gain gain = Gain.INFO;

    private final ClassSet classSet;

    private final Distance metric;
    private final int inspectedShapelets;
    private final double aggregateFraction;
    private final double minSplit;
    private final SampleMode sampleMode;
    private final Assessment assessment;
    private double lowerLength;
    private double upperLength;
    private Vector classes;

    protected Learner() {
      this(new Configurator(), null, null);
    }

    public Learner(ClassSet classSet, Vector classes) {
      this(new Configurator(), classSet, classes);
    }

    protected Learner(Configurator builder, ClassSet classSet, Vector classes) {
      this.metric = builder.metric;
      this.inspectedShapelets = builder.inspectedShapelets;
      this.lowerLength = builder.lowerLength;
      this.upperLength = builder.upperLength;
      this.aggregateFraction = builder.aggregateFraction;
      this.sampleMode = builder.sampleMode;
      this.assessment = builder.assessment;
      this.minSplit = builder.minSplit;

      Check.inRange(upperLength, lowerLength, 1);
      Check.inRange(lowerLength, 0, upperLength);
      this.classSet = classSet;
      this.classes = classes;
    }

    public Learner(double low, double high, Configurator builder, ClassSet sample, Vector classes) {
      this(builder, sample, classes);
      this.lowerLength = low;
      this.upperLength = high;
    }

    public Random getRandom() {
      return random;
    }

    public Gain getGain() {
      return gain;
    }

    public Distance getDistanceMetric() {
      return metric;
    }

    public int getInspectedShapelets() {
      return inspectedShapelets;
    }

    public double getLowerLength() {
      return lowerLength;
    }

    public double getUpperLength() {
      return upperLength;
    }

    @Override
    public ShapeletTree fit(DataFrame x, Vector y) {
      ClassSet classSet = this.classSet;
      Vector classes = this.classes != null ? this.classes : Vectors.unique(y);
      if (classSet == null) {
        classSet = new ClassSet(y, classes);
      }

      DataFrame dataFrame = x;
      if (sampleMode == SampleMode.DOWN_SAMPLE) {
        Check.inRange(aggregateFraction, 0.1, 1);
        dataFrame = Approximations.paa(x, (int) Math.round(x.columns() * aggregateFraction));
      }

      Params params = new Params();
      params.noExamples = classSet.getTotalWeight();
      params.lengthImportance = DoubleArray.zeros(x.columns());
      params.positionImportance = DoubleArray.zeros(x.columns());
      params.originalData = x;
      params.shapeStore = new ShapeStore();
      TreeNode<ShapeletThreshold> node = build(dataFrame, y, classSet, params);
      /* new ShapletTreeVisitor(size, getDistanceMetric()) */
      return new ShapeletTree(classes, node, new ShapeletTree.Learner.ShapletTreeVisitor(10,
          getDistanceMetric()), params.lengthImportance, params.positionImportance, params.depth,
          classSet, params.shapeStore);
      // return new ShapeletTree(classes, node, new WeightVisitor(getDistanceMetric()),
      // params.lengthImportance, params.positionImportance, params.depth, classSet);
      // return new Classifier(classes, node, new OneNnVisitor(getDistanceMetric(), x, y),
      // params.lengthImportance, params.positionImportance, params.depth, classSet);
      // return new Classifier(
      // classes, node, new GuessVisitor(getDistanceMetric()),
      // params.lengthImportance, params.positionImportance, params.depth, classSet
      // );
    }

    protected TreeNode<ShapeletThreshold> build(DataFrame x, Vector y, ClassSet classSet,
        Params params) {
      if (classSet.getTotalWeight() <= minSplit || classSet.getTargetCount() == 1) {
        return TreeLeaf.fromExamples(classSet, classSet.getTotalWeight() / params.noExamples);
      }
      params.depth += 1;
      TreeSplit<ShapeletThreshold> maxSplit = find(classSet, x, y, params);
      if (maxSplit == null) {
        return TreeLeaf.fromExamples(classSet, classSet.getTotalWeight() / params.noExamples);
      } else {
        ClassSet left = maxSplit.getLeft();
        ClassSet right = maxSplit.getRight();
        if (left.isEmpty()) {
          return TreeLeaf.fromExamples(right, right.getTotalWeight() / params.noExamples);
        } else if (right.isEmpty()) {
          return TreeLeaf.fromExamples(left, left.getTotalWeight() / params.noExamples);
        } else {
          Shapelet shapelet = maxSplit.getThreshold().getShapelet();
          Impurity impurity = getGain().getImpurity();
          double imp = impurity.impurity(classSet);
          double weight = (maxSplit.size() / params.noExamples) * (imp - maxSplit.getImpurity());

          params.lengthImportance.set(shapelet.size(), params.lengthImportance.get(shapelet.size())
              + weight);
          int length = shapelet.size();
          int start = shapelet.start();
          int end = start + length;
          for (int i = start; i < end; i++) {
            params.positionImportance.set(i, params.positionImportance.get(i) + (weight / length));
          }

          params.shapeStore.add(shapelet, weight);

          TreeNode<ShapeletThreshold> leftNode = build(x, y, left, params);
          TreeNode<ShapeletThreshold> rightNode = build(x, y, right, params);
          Vector.Builder classDist = Vector.Builder.of(double.class);
          for (Object target : classSet.getTargets()) {
            classDist.set(target, classSet.get(target).getWeight());
          }

          return new TreeBranch<>(leftNode, rightNode, classes, classDist.build(),
              maxSplit.getThreshold(), classSet.getTotalWeight() / params.noExamples);
        }
      }
    }

    public TreeSplit<ShapeletThreshold> find(ClassSet classSet, DataFrame x, Vector y, Params params) {
      return getUnivariateShapeletThreshold(classSet, x, y, params);
    }

    protected TreeSplit<ShapeletThreshold> getUnivariateShapeletThreshold(ClassSet classSet,
        DataFrame x, Vector y, Params params) {
      int maxShapelets = this.inspectedShapelets;
      if (maxShapelets < 0) {
        maxShapelets = maxShapelets(x.columns());
      }

      // TODO: add alternative shapelet sampling approaches.
      // The simple approach to ddo this is to add shapelets to the list `shapelets` below.
      // Shapelet are created with the new IndexSortedNormalizedShapelet(vector, start, end);
      // constructor.
      List<Shapelet> shapelets = new ArrayList<>(maxShapelets);
      if (sampleMode == SampleMode.NEW_SAMPLE) {
        // NOTE: This is dead code and do not work as expected
        int n = x.rows();
        int m = x.columns();
        Random rand = random;
        double sum = 0;
        for (int i = 3; i <= m; i++) {
          sum += m - i + 1;
        }
        double f = (aggregateFraction * m) / sum;
        if (Math.round(f * m) < 1) {
          throw new IllegalArgumentException("Won't generate any shapelets.");
        }
        for (int i = 3; i <= m; i++) {
          long r = Math.round(f * (m - i + 1));
          for (int j = 0; j < r; j++) {
            int vec = classSet.getRandomSample().getRandomExample().getIndex();
            int start = rand.nextInt(m + 1 - i);
            shapelets.add(new IndexSortedNormalizedShapelet(start, i, x.loc().getRecord(vec)));
          }
        }
        System.out.println(shapelets.size());
      } else {
        for (int i = 0; i < maxShapelets; i++) {
          int index = classSet.getRandomSample().getRandomExample().getIndex();
          Vector timeSeries = x.loc().getRecord(index);
          Object shapelet;
          if (Vector.class.isAssignableFrom(timeSeries.getType().getDataClass())) {
            int channelIndex = random.nextInt(timeSeries.size());
            Vector channel = timeSeries.loc().get(Vector.class, channelIndex);
            Shapelet univariateShapelet = getUnivariateShapelet(classSet, x, index, channel);
            if (univariateShapelet != null) {
              shapelet = new ChannelShapelet(channelIndex, univariateShapelet);
            } else {
              shapelet = null;
            }
          } else {
            shapelet = getUnivariateShapelet(classSet, x, index, timeSeries);
          }
          if (shapelet == null) {
            continue;
          }
          if (shapelet instanceof List) {
            @SuppressWarnings("unchecked")
            List<Shapelet> shapeletList = (List<Shapelet>) shapelet;
            shapelets.addAll(shapeletList);
          } else {
            shapelets.add((Shapelet) shapelet);
          }
        }
      }

      if (shapelets.isEmpty()) {
        return null;
      }

      TreeSplit<ShapeletThreshold> bestSplit;
      if (assessment == Assessment.IG) {
        bestSplit = findBestSplit(classSet, x, y, shapelets);
      } else {
        bestSplit = findBestSplitFstat(classSet, x, y, shapelets);
      }

      if (sampleMode == SampleMode.DOWN_SAMPLE) {
        DownsampledShapelet best = (DownsampledShapelet) bestSplit.getThreshold().getShapelet();
        Vector record = params.originalData.loc().getRecord(best.index);
        Shapelet shapelet = new IndexSortedNormalizedShapelet(best.start, best.length, record);
        return findBestSplit(classSet, params.originalData, y, Collections.singletonList(shapelet));
      } else {
        return bestSplit;
      }
    }

    private int maxShapelets(int columns) {
      return (int) Math.round(Math.sqrt(columns * (columns + 1) / 2));
    }

    private Shapelet getUnivariateShapelet(ClassSet classSet, DataFrame x, int index,
        Vector timeSeries) {
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
        return null;
      }

      int length = random.nextInt(upper) + lower;
      int start = random.nextInt(timeSeriesLength - length);
      Shapelet shapelet;
      if (sampleMode == SampleMode.DOWN_SAMPLE) {
        shapelet = getDownsampledShapelet(index, timeSeries, timeSeriesLength, length, start);
      } else if (sampleMode == SampleMode.RANDOMIZE) {
        shapelet = getRandomizedShapelet(classSet, x, length, start);
      } else if (sampleMode == SampleMode.DERIVATE
          && ThreadLocalRandom.current().nextGaussian() > 0) {
        shapelet = getDerivativeShapelet(timeSeries, timeSeriesLength, length, start);
      } else {
        shapelet = new IndexSortedNormalizedShapelet(start, length, timeSeries);
      }
      return shapelet;
    }

    private Shapelet getDerivativeShapelet(Vector timeSeries, int timeSeriesLength, int length,
        int start) {
      Vector.Builder derivative = Vector.Builder.withCapacity(Double.class, timeSeriesLength);
      derivative.loc().set(0, 0);
      for (int j = 1; j < timeSeriesLength; j++) {
        derivative.loc().set(j,
            timeSeries.loc().getAsDouble(j) - timeSeries.loc().getAsDouble(j - 1));
      }
      return new DerivativeShapelet(start, length, derivative.build());
    }

    private Shapelet getRandomizedShapelet(ClassSet classSet, DataFrame x, int length, int start) {
      Vector.Builder meanVec = Vector.Builder.of(Double.class);
      for (int j = 0; j < 10; j++) {
        Vector record = x.loc().getRecord(classSet.getRandomSample().getRandomExample().getIndex());
        Shapelet shapelet = new Shapelet(start, length, record);
        for (int k = 0; k < shapelet.size(); k++) {
          meanVec.set(k, shapelet.loc().getAsDouble(k) / 10);
        }
      }
      return new IndexSortedNormalizedShapelet(0, meanVec.size(), meanVec.build());
    }

    private Shapelet getDownsampledShapelet(int index, Vector timeSeries, int timeSeriesLength,
        int length, int start) {
      int downStart = (int) Math.round(start * aggregateFraction);
      int downLength = (int) Math.round(length * aggregateFraction);
      if (downStart + downLength > timeSeriesLength * aggregateFraction) {
        downLength -= 1;
      }
      return new DownsampledShapelet(index, start, length, downStart, downLength, timeSeries);
    }

    protected TreeSplit<ShapeletThreshold> findBestSplit(ClassSet classSet, DataFrame x, Vector y,
        List<Shapelet> shapelets) {
      TreeSplit<ShapeletThreshold> bestSplit = null;
      Threshold bestThreshold = Threshold.inf();
      IntDoubleMap bestDistanceMap = null;
      for (Shapelet shapelet : shapelets) {
        IntDoubleMap distanceMap = new IntDoubleOpenHashMap();
        Threshold threshold = bestDistanceThresholdInSample(classSet, x, y, shapelet, distanceMap);
        boolean lowerImpurity = threshold.impurity < bestThreshold.impurity;
        boolean equalImpuritySmallerGap =
            threshold.impurity == bestThreshold.impurity && threshold.gap > bestThreshold.gap;
        if (lowerImpurity || equalImpuritySmallerGap) {
          bestSplit = split(distanceMap, classSet, threshold.threshold, shapelet);
          bestThreshold = threshold;
          bestDistanceMap = distanceMap;
        }
      }

      if (bestSplit != null) {
        bestSplit.setImpurity(bestThreshold.impurity);
        ShapeletThreshold threshold = bestSplit.getThreshold();
        threshold.setClassDistances(computeMeanDistance(bestDistanceMap, classSet, y));
      }
      return bestSplit;
    }

    private Vector computeMeanDistance(IntDoubleMap bestDistanceMap, ClassSet classSet, Vector y) {
      Map<Object, FastStatistics> cmd = new HashMap<>();
      for (Example example : classSet) {
        double distance = bestDistanceMap.get(example.getIndex());
        Object cls = y.loc().get(Object.class, example.getIndex());
        FastStatistics statistics = cmd.get(cls);
        if (statistics == null) {
          statistics = new FastStatistics();
          cmd.put(cls, statistics);
        }
        statistics.addValue(distance);
      }
      Vector.Builder builder = Vector.Builder.of(double.class);
      for (Map.Entry<Object, FastStatistics> entry : cmd.entrySet()) {
        builder.set(entry.getKey(), entry.getValue().getMean());
      }
      return builder.build();
    }

    protected Threshold bestDistanceThresholdInSample(ClassSet classSet, DataFrame x, Vector y,
        Shapelet shapelet, IntDoubleMap memoizedDistances) {
      double sum = 0.0;
      List<ExampleDistance> distances = new ArrayList<>();
      Distance distanceMetric = getDistanceMetric();
      for (Example example : classSet) {
        Vector record = x.loc().getRecord(example.getIndex());
        double distance;
        if (shapelet instanceof ChannelShapelet) {
          Vector channel =
              record.loc().get(Vector.class, ((ChannelShapelet) shapelet).getChannel());
          distance = distanceMetric.compute(channel, shapelet);
        } else {
          distance = distanceMetric.compute(record, shapelet);
        }
        memoizedDistances.put(example.getIndex(), distance);
        distances.add(new ExampleDistance(distance, example));
        sum += distance;
      }

      Collections.sort(distances);
      return findBestThreshold(distances, classSet, y, sum);
    }

    protected TreeSplit<ShapeletThreshold> findBestSplitFstat(ClassSet classSet, DataFrame x,
        Vector y, List<Shapelet> shapelets) {
      IntDoubleMap bestDistanceMap = null;
      List<ExampleDistance> bestDistances = null;
      double bestStat = Double.NEGATIVE_INFINITY;
      Shapelet bestShapelet = null;
      double bestSum = 0;

      Distance metric = getDistanceMetric();
      for (Shapelet shapelet : shapelets) {
        List<ExampleDistance> distances = new ArrayList<>();
        IntDoubleMap distanceMap = new IntDoubleOpenHashMap();
        double sum = 0;
        for (Example example : classSet) {
          Vector record = x.loc().getRecord(example.getIndex());
          double dist;
          if (shapelet instanceof ChannelShapelet) {
            Vector channel =
                record.loc().get(Vector.class, ((ChannelShapelet) shapelet).getChannel());
            dist = metric.compute(channel, shapelet);
          } else {
            dist = metric.compute(record, shapelet);
          }

          distanceMap.put(example.getIndex(), dist);
          distances.add(new ExampleDistance(dist, example));
          sum += dist;
        }
        double stat = assessFstatShapeletQuality(distances, y);
        // TODO: comment away
        // stat *= (shapelet.size() / (double) x.columns());
        if (stat > bestStat || bestDistances == null) {
          bestStat = stat;
          bestDistanceMap = distanceMap;
          bestDistances = distances;
          bestShapelet = shapelet;
          bestSum = sum;
        }
      }

      Threshold t = findBestThreshold(bestDistances, classSet, y, bestSum);
      TreeSplit<ShapeletThreshold> split =
          split(bestDistanceMap, classSet, t.threshold, bestShapelet);
      split.setImpurity(t.impurity);
      return split;
    }

    private double assessFstatShapeletQuality(List<ExampleDistance> distances, Vector y) {
      ObjectDoubleMap<Object> sums = new ObjectDoubleOpenHashMap<>();
      ObjectDoubleMap<Object> sumsSquared = new ObjectDoubleOpenHashMap<>();
      ObjectDoubleMap<Object> sumOfSquares = new ObjectDoubleOpenHashMap<>();
      ObjectIntMap<Object> sizes = new ObjectIntOpenHashMap<>();

      int numInstances = distances.size();
      for (ExampleDistance distance : distances) {
        Object c = y.loc().get(Object.class, distance.example.getIndex()); // getClassVal
        double thisDist = distance.distance; // getDistance
        sizes.addTo(c, 1);
        sums.addTo(c, thisDist); // sums[c] += thisDist
        sumOfSquares.addTo(c, thisDist * thisDist); // sumsOfSquares[c] += thisDist + thisDist
      }
      //
      double part1 = 0;
      double part2 = 0;
      for (ObjectDoubleCursor<Object> sum : sums) {
        sumsSquared.put(sum.key, sum.value * sum.value); // sumsSquared[i] = sums[i] * sums[i]
        part1 += sumOfSquares.get(sum.key); // sumOfSquares[i]
        part2 += sum.value; // sums[i]
      }
      part2 *= part2;
      part2 /= numInstances;
      double ssTotal = part1 - part2;

      part1 = 0;
      part2 = 0;
      for (ObjectDoubleCursor<Object> c : sumsSquared) {
        part1 += c.value / sizes.get(c.key);
        part2 += sums.get(c.key);
      }
      double ssAmong = part1 - (part2 * part2) / numInstances;
      double ssWithin = ssTotal - ssAmong;
      int dfAmong = sums.size() - 1;
      int dfWithin = numInstances - sums.size();
      double msAmong = ssAmong / dfAmong;
      double msWithin = ssWithin / dfWithin;
      double f = msAmong / msWithin;
      return Double.isNaN(f) ? 0 : f;
    }

    public Threshold findBestThreshold(List<ExampleDistance> distances, ClassSet classSet,
        Vector y, double distanceSum) {
      ObjectDoubleMap<Object> lt = new ObjectDoubleOpenHashMap<>();
      ObjectDoubleMap<Object> gt = new ObjectDoubleOpenHashMap<>();

      List<Object> presentTargets = classSet.getTargets();
      DoubleArray ltRelativeFrequency = DoubleArray.zeros(presentTargets.size());
      DoubleArray gtRelativeFrequency = DoubleArray.zeros(presentTargets.size());

      double ltWeight = 0.0, gtWeight = 0.0;

      // Initialize all value to the right (i.e. all values are larger than the initial threshold)
      for (ClassSet.Sample sample : classSet.samples()) {
        double weight = sample.getWeight();
        gtWeight += weight;

        lt.put(sample.getTarget(), 0);
        gt.put(sample.getTarget(), weight);
      }

      // Transfer weights from the initial example
      Example first = distances.get(0).example;
      Object prevTarget = y.loc().get(Object.class, first.getIndex());
      gt.addTo(prevTarget, -first.getWeight());
      lt.addTo(prevTarget, first.getWeight());
      gtWeight -= first.getWeight();
      ltWeight += first.getWeight();

      double prevDistance = distances.get(0).distance;
      double lowestImpurity = Double.POSITIVE_INFINITY;
      double threshold = distances.get(0).distance / 2;
      Gain gain = getGain();
      double ltGap = 0.0, gtGap = distanceSum, largestGap = Double.NEGATIVE_INFINITY;
      for (int i = 1; i < distances.size(); i++) {
        ExampleDistance ed = distances.get(i);
        Object target = y.loc().get(Object.class, ed.example.getIndex());

        // IF previous target NOT EQUALS current target and the previous distance equals the current
        // (except for the first)
        boolean notSameDistance = ed.distance != prevDistance;
        boolean firstOrEqualTarget = prevTarget == null || !prevTarget.equals(target);
        boolean firstIteration = i == 1;
        if (firstIteration || notSameDistance && firstOrEqualTarget) {

          // Generate the relative frequency distribution
          for (int j = 0; j < presentTargets.size(); j++) {
            Object presentTarget = presentTargets.get(j);
            ltRelativeFrequency.set(j, ltWeight != 0 ? lt.get(presentTarget) / ltWeight : 0);
            gtRelativeFrequency.set(j, gtWeight != 0 ? gt.get(presentTarget) / gtWeight : 0);
          }

          // If this split is better, update the threshold
          double impurity =
              gain.compute(ltWeight, ltRelativeFrequency, gtWeight, gtRelativeFrequency);
          double gap = (1 / ltWeight * ltGap) - (1 / gtWeight * gtGap);
          boolean lowerImpurity = impurity < lowestImpurity;
          boolean equalImpuritySmallerGap = impurity == lowestImpurity && gap > largestGap;
          if (lowerImpurity || equalImpuritySmallerGap) {
            lowestImpurity = impurity;
            largestGap = gap;
            threshold = (ed.distance + prevDistance) / 2;
          }
        }

        /*
         * Move cursor one example forward, and adjust the weights accordingly. Then calculate the
         * new gain for moving the threshold. If this results in a cleaner split, adjust the
         * threshold (by taking the average of the current and the previous value).
         */
        double weight = ed.example.getWeight();
        ltWeight += weight;
        gtWeight -= weight;
        lt.addTo(target, weight);
        gt.addTo(target, -weight);

        ltGap += ed.distance;
        gtGap -= ed.distance;

        prevDistance = ed.distance;
        prevTarget = target;
      }

      double minimumMargin = Double.POSITIVE_INFINITY;
      return new Threshold(threshold, lowestImpurity, largestGap, minimumMargin);
    }

    protected TreeSplit<ShapeletThreshold> split(IntDoubleMap distanceMap, ClassSet classSet,
        double threshold, Shapelet shapelet) {
      ClassSet left = new ClassSet(classSet.getDomain());
      ClassSet right = new ClassSet(classSet.getDomain());
      for (ClassSet.Sample sample : classSet.samples()) {
        Object target = sample.getTarget();

        ClassSet.Sample leftSample = ClassSet.Sample.create(target);
        ClassSet.Sample rightSample = ClassSet.Sample.create(target);

        for (Example example : sample) {
          double shapeletDistance = distanceMap.get(example.getIndex());
          if (shapeletDistance == threshold) {
            if (getRandom().nextDouble() <= 0.5) {
              leftSample.add(example);
            } else {
              rightSample.add(example);
            }
          } else if (shapeletDistance < threshold) {
            leftSample.add(example);
          } else {
            rightSample.add(example);
          }
        }

        if (!leftSample.isEmpty()) {
          left.add(leftSample);
        }
        if (!rightSample.isEmpty()) {
          right.add(rightSample);
        }
      }

      return new TreeSplit<>(left, right, new ShapeletThreshold(shapelet, threshold, classSet));
    }

    public enum SampleMode {
      DOWN_SAMPLE, NORMAL, DERIVATE, NEW_SAMPLE, RANDOMIZE
    }

    public enum Assessment {
      IG, FSTAT
    }

    private static class ExampleDistance implements Comparable<ExampleDistance> {

      public final double distance;
      public final Example example;

      public ExampleDistance(double distance, Example example) {
        this.distance = distance;
        this.example = example;
      }

      @Override
      public int compareTo(ExampleDistance o) {
        return Double.compare(distance, o.distance);
      }

      @Override
      public String toString() {
        return String.format("ExampleDistance(id=%d, %.2f)", example.getIndex(), distance);
      }
    }

    private static class DownsampledShapelet extends IndexSortedNormalizedShapelet {

      private final int start;
      private final int length;
      private final int index;

      public DownsampledShapelet(int index, int start, int length, int downStart, int downLength,
          Vector vector) {
        super(downStart, downLength, vector);

        this.start = start;
        this.length = length;
        this.index = index;
      }
    }

    protected static class Threshold {

      public double threshold, impurity, gap, margin;

      public Threshold(double threshold, double impurity, double gap, double margin) {
        this.threshold = threshold;
        this.impurity = impurity;
        this.gap = gap;
        this.margin = margin;
      }

      public static Threshold inf() {
        return new Threshold(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
            Double.NEGATIVE_INFINITY);
      }

      public boolean isBetterThan(Threshold bestThreshold) {
        return this.impurity < bestThreshold.impurity
            || (this.impurity == bestThreshold.impurity && this.gap > bestThreshold.gap);
      }
    }

    private static class Params {

      public double noExamples;
      public DataFrame originalData;
      private DoubleArray lengthImportance;
      private DoubleArray positionImportance;
      private int depth = 0;
      public ShapeStore shapeStore;
    }

    private static class GuessVisitor implements TreeVisitor<ShapeletThreshold> {

      private final Distance distanceMeasure;

      private GuessVisitor(Distance distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
      }

      @Override
      public DoubleArray visitLeaf(TreeLeaf<ShapeletThreshold> leaf, Vector example) {
        return leaf.getProbabilities();
      }

      @Override
      public DoubleArray visitBranch(TreeBranch<ShapeletThreshold> node, Vector example) {
        Shapelet shapelet = node.getThreshold().getShapelet();
        double threshold = node.getThreshold().getDistance();
        if (shapelet.size() > example.size()) {
          Vector mcd = node.getThreshold().getClassDistances();
          double d = distanceMeasure.compute(shapelet, example);
          double sqrt = Math.sqrt((d * d * example.size()) / shapelet.size());
          // if (sqrt < threshold) {
          // return visit(node.getLeft(), example);
          // } else {
          // return visit(node.getRight(), example);
          // }
          //
          double min = Double.POSITIVE_INFINITY;
          Object minKey = null;
          for (Object key : mcd.getIndex()) {
            double dist = Math.abs(sqrt - mcd.getAsDouble(key));
            if (dist < min) {
              min = dist;
              minKey = key;
            }
          }

          double left =
              node.getLeft().getClassDistribution().getAsDouble(minKey, Double.NEGATIVE_INFINITY);
          double right =
              node.getRight().getClassDistribution().getAsDouble(minKey, Double.NEGATIVE_INFINITY);
          if (left > right) {
            return visit(node.getLeft(), example);
          } else {
            return visit(node.getRight(), example);
          }
          // DoubleArray doubleArray = Arrays.doubleArray(node.getDomain().size());
          // for (int i = 0; i < node.getDomain().size(); i++) {
          // if (minKey.equals(node.getDomain().loc().get(Object.class, i))) {
          // doubleArray.set(i, 1);
          // break;
          // }
          // }
          // return doubleArray;
        } else {
          double computedDistance;
          if (shapelet instanceof ChannelShapelet) {
            int shapeletChannel = ((ChannelShapelet) shapelet).getChannel();
            Vector exampleChannel = example.loc().get(Vector.class, shapeletChannel);
            computedDistance = distanceMeasure.compute(exampleChannel, shapelet);
          } else {
            computedDistance = distanceMeasure.compute(example, shapelet);
          }
          if (computedDistance < threshold) {
            return visit(node.getLeft(), example);
          } else {
            return visit(node.getRight(), example);
          }
        }
      }
    }

    private static class OneNnVisitor implements TreeVisitor<ShapeletThreshold> {

      public static final EuclideanDistance EUCLIDEAN = EuclideanDistance.getInstance();
      private final Distance distanceMeasure;
      private final DataFrame x;
      private final Vector y;

      private OneNnVisitor(Distance distanceMeasure, DataFrame x, Vector y) {
        this.distanceMeasure = distanceMeasure;
        this.x = x;
        this.y = y;
      }


      @Override
      public DoubleArray visitLeaf(TreeLeaf<ShapeletThreshold> leaf, Vector example) {
        return leaf.getProbabilities();
      }

      @Override
      public DoubleArray visitBranch(TreeBranch<ShapeletThreshold> node, Vector example) {
        Shapelet shapelet = node.getThreshold().getShapelet();
        double threshold = node.getThreshold().getDistance();
        if (shapelet.size() >= example.size()) {
          // Use 1NN
          ClassSet included = node.getThreshold().getClassSet();
          double minDistance = Double.POSITIVE_INFINITY;
          Object cls = null;
          for (Example ex : included) {
            double distance = EUCLIDEAN.compute(example, x.loc().getRecord(ex.getIndex()));
            if (distance < minDistance) {
              minDistance = distance;
              cls = y.loc().get(Object.class, ex.getIndex());
            }
          }
          double left =
              node.getLeft().getClassDistribution().getAsDouble(cls, Double.NEGATIVE_INFINITY);
          double right =
              node.getRight().getClassDistribution().getAsDouble(cls, Double.NEGATIVE_INFINITY);
          if (left > right) {
            return visit(node.getLeft(), example);
          } else {
            return visit(node.getRight(), example);
          }
          // DoubleArray doubleArray = Arrays.doubleArray(node.getDomain().size());
          // for (int i = 0; i < node.getDomain().size(); i++) {
          // if (cls.equals(node.getDomain().loc().get(Object.class, i))) {
          // doubleArray.set(i, 1);
          // break;
          // }
          // }
          // return doubleArray;
        } else {
          double computedDistance;
          if (shapelet instanceof ChannelShapelet) {
            int shapeletChannel = ((ChannelShapelet) shapelet).getChannel();
            Vector exampleChannel = example.loc().get(Vector.class, shapeletChannel);
            computedDistance = distanceMeasure.compute(exampleChannel, shapelet);
          } else {
            computedDistance = distanceMeasure.compute(example, shapelet);
          }
          if (computedDistance < threshold) {
            return visit(node.getLeft(), example);
          } else {
            return visit(node.getRight(), example);
          }
        }
      }
    }


    private static class WeightVisitor implements TreeVisitor<ShapeletThreshold> {

      private final Distance distanceMeasure;
      private final double weight;

      private WeightVisitor(Distance distanceMeasure, double weight) {
        this.distanceMeasure = distanceMeasure;
        this.weight = weight;
      }

      public WeightVisitor(Distance distanceMeasure) {
        this(distanceMeasure, 1);
      }

      @Override
      public DoubleArray visitLeaf(TreeLeaf<ShapeletThreshold> leaf, Vector example) {
        return leaf.getProbabilities().times(weight);
      }

      @Override
      public DoubleArray visitBranch(TreeBranch<ShapeletThreshold> node, Vector example) {
        Shapelet shapelet = node.getThreshold().getShapelet();
        double threshold = node.getThreshold().getDistance();
        if (shapelet.size() > example.size()) {
          // if (shapelet.start() < example.size()) {
          // WeightVisitor visitor = new WeightVisitor(distanceMeasure, weight);
          // int size = Math.min(shapelet.size(), example.size());
          // double residual = 0.0;
          // for (int i = 0; i < size; i++) {
          // double r = shapelet.loc().getAsDouble(i) - example.loc().getAsDouble(i);
          // residual += r * r;
          // }
          // double d2 = Math.sqrt(residual / shapelet.size());
          // if (d2 < threshold) {
          // return visitor.visit(node.getLeft(), example);
          // } else {
          // return visitor.visit(node.getRight(), example);
          // }
          // } else {
          WeightVisitor leftVisitor =
              new WeightVisitor(distanceMeasure, node.getLeft().getWeight());
          WeightVisitor rightVisitor =
              new WeightVisitor(distanceMeasure, node.getRight().getWeight());
          DoubleArray leftProbabilities = leftVisitor.visit(node.getLeft(), example);
          DoubleArray rightProbabilities = rightVisitor.visit(node.getRight(), example);
          return leftProbabilities.plus(rightProbabilities);
          // }
        } else {
          WeightVisitor visitor = new WeightVisitor(distanceMeasure, weight);
          double computedDistance;
          if (shapelet instanceof ChannelShapelet) {
            ChannelShapelet channelShapelet = (ChannelShapelet) shapelet;
            Vector channel = example.loc().get(Vector.class, channelShapelet.getChannel());
            computedDistance = distanceMeasure.compute(channel, shapelet);
          } else {
            computedDistance = distanceMeasure.compute(example, shapelet);
          }
          if (computedDistance < threshold) {
            return visitor.visit(node.getLeft(), example);
          } else {
            return visitor.visit(node.getRight(), example);
          }
        }
      }
    }

    private static class ShapletTreeVisitor implements TreeVisitor<ShapeletThreshold> {

      private final Distance distanceMeasure;
      private final Aggregator aggregator;

      private ShapletTreeVisitor(int size, Distance distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
        this.aggregator = new MeanAggregator(size);
      }

      @Override
      public DoubleArray visitLeaf(TreeLeaf<ShapeletThreshold> leaf, Vector example) {
        return leaf.getProbabilities();
      }

      @Override
      public DoubleArray visitBranch(TreeBranch<ShapeletThreshold> node, Vector example) {
        Shapelet shapelet = node.getThreshold().getShapelet();
        double threshold = node.getThreshold().getDistance();
        // if (shapelet.size() > example.size()) {
        // double d = distanceMeasure.compute(shapelet, example);
        // double sqrt = Math.sqrt((d * d) / shapelet.size());
        // if (sqrt < threshold) {
        // return visit(node.getLeft(), example);
        // } else {
        // return visit(node.getRight(), example);
        // }
        // } else {
        double computedDistance;
        if (shapelet instanceof ChannelShapelet) {
          int shapeletChannel = ((ChannelShapelet) shapelet).getChannel();
          Vector exampleChannel = example.loc().get(Vector.class, shapeletChannel);
          computedDistance = distanceMeasure.compute(exampleChannel, shapelet);
        } else {
          computedDistance = distanceMeasure.compute(example, shapelet);
        }
        if (computedDistance < threshold) {
          return visit(node.getLeft(), example);
        } else {
          return visit(node.getRight(), example);
        }
        // }
      }
    }
  }

  public static class Configurator implements Classifier.Configurator<Learner> {

    public Learner.Assessment assessment = Learner.Assessment.FSTAT;
    public double minSplit = 1;
    public Distance metric = EarlyAbandonSlidingDistance.create(EuclideanDistance.getInstance());
    public int inspectedShapelets = 100;
    public double aggregateFraction = 1;
    public Learner.SampleMode sampleMode = Learner.SampleMode.NORMAL;
    public double lowerLength = 0.01;
    public double upperLength = 1;

    public Configurator() {}

    public Classifier.Configurator setMinimumSplit(double minSplit) {
      this.minSplit = minSplit;
      return this;
    }

    public Classifier.Configurator setDistance(Distance metric) {
      this.metric = metric;
      return this;

    }

    public Classifier.Configurator setAssessment(Learner.Assessment assessment) {
      this.assessment = assessment;
      return this;
    }

    public Classifier.Configurator setMaximumShapelets(int inspectedShapelets) {
      this.inspectedShapelets = inspectedShapelets;
      return this;

    }

    public Classifier.Configurator setLowerLength(double lowerLength) {
      this.lowerLength = lowerLength;
      return this;
    }

    public Classifier.Configurator setUpperLength(double upperLength) {
      this.upperLength = upperLength;
      return this;

    }

    public Classifier.Configurator setSampleMode(Learner.SampleMode sampleMode) {
      this.sampleMode = sampleMode;
      return this;
    }

    public Classifier.Configurator setAggregateFraction(double aggregateFraction) {
      this.aggregateFraction = aggregateFraction;
      return this;
    }

    public Learner configure() {
      return new Learner();
    }
  }
}
