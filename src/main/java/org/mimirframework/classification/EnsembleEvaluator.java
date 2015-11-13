package org.mimirframework.classification;

import static org.briljantframework.array.Arrays.argmax;
import static org.briljantframework.data.vector.Vectors.find;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.evaluation.EvaluationContext;
import org.mimirframework.evaluation.Evaluator;
import org.mimirframework.evaluation.MeasureCollection;

/**
 * Created by isak on 11/13/15.
 */
public enum EnsembleEvaluator implements Evaluator<Ensemble> {

  INSTANCE;

  private static void computeOOBCorrelation(EvaluationContext<? extends Ensemble> ctx) {
    Ensemble ensemble = ctx.getPredictor();
    Vector classes = ensemble.getClasses();
    DataFrame x = ctx.getPartition().getTrainingData();
    Vector y = ctx.getPartition().getTrainingTarget();

    BooleanArray oobIndicator = ensemble.getOobIndicator();
    List<Classifier> members = ensemble.getEnsembleMembers();
    MeasureCollection measureCollection = ctx.getMeasureCollection();

    // Store the out-of-bag and in-bag probability estimates
    DoubleArray oobEstimates = DoubleArray.zeros(x.rows(), classes.size());
    DoubleArray inbEstimates = DoubleArray.zeros(x.rows(), classes.size());

    // Count the number of times each training sample have been included
    IntArray counts = Arrays.sum(1, oobIndicator.asInt());

    // Compute the in-bag and out-of-bag estimates for all examples
    DoubleAdder oobAccuracy = new DoubleAdder();
    IntStream.range(0, x.rows()).parallel().forEach(i -> {
      int inbSize = members.size() - counts.get(i);
      int oobSize = counts.get(i);
      Vector record = x.loc().getRecord(i);
      for (int j = 0; j < members.size(); j++) {
        DoubleArray estimate = members.get(j).estimate(record);
        if (oobIndicator.get(i, j)) {
          oobEstimates.getRow(i).assign(estimate, (e, v) -> e + v / oobSize);
        } else {
          inbEstimates.getRow(i).assign(estimate, (e, v) -> e + v / inbSize);
        }
      }
      oobAccuracy.add(find(classes, y, i) == argmax(oobEstimates.getRow(i)) ? 1 : 0);
    });
    double avgOobAccuracy = oobAccuracy.sum() / x.rows();
    measureCollection.add("oobError", 1 - avgOobAccuracy);
    // ctx.getOrDefault(OobAccuracy.class, OobAccuracy.Builder::new).add(OUT, avgOobAccuracy);

    DoubleAdder strengthA = new DoubleAdder();
    DoubleAdder strengthSquareA = new DoubleAdder();
    IntStream.range(0, oobEstimates.rows()).parallel().forEach(i -> {
      DoubleArray estimation = oobEstimates.getRow(i);
      int c = find(classes, y, i);
      double ma = estimation.get(c) - maxnot(estimation, c);
      strengthA.add(ma);
      strengthSquareA.add(ma * ma);
    });

    double strength = strengthA.doubleValue() / y.size();
    double strengthSquare = strengthSquareA.doubleValue() / y.size();
    double s2 = strength * strength;
    double variance = strengthSquare - s2;
    double std = 0;
    for (int j = 0; j < members.size(); j++) {
      Classifier member = members.get(j);
      AtomicInteger oobSizeA = new AtomicInteger(0);
      DoubleAdder p1A = new DoubleAdder();
      DoubleAdder p2A = new DoubleAdder();
      final int memberIndex = j;
      IntStream.range(0, x.rows()).parallel().forEach(i -> {
        if (oobIndicator.get(i, memberIndex)) {
          oobSizeA.getAndIncrement();
          int c = find(classes, y, i);
          DoubleArray memberEstimation = member.estimate(x.loc().getRecord(i));
          DoubleArray ibEstimation = inbEstimates.getRow(i);
          p1A.add(argmax(memberEstimation) == c ? 1 : 0);
          p2A.add(argmax(memberEstimation) == argmaxnot(ibEstimation, c) ? 1 : 0);
        }
      });
      double p1 = p1A.sum() / oobSizeA.get();
      double p2 = p2A.sum() / oobSizeA.get();
      std += Math.sqrt(p1 + p2 + Math.pow(p1 - p2, 2));
    }
    std = Math.pow(std / members.size(), 2);
    double correlation = variance / std;
    double errorBound = (correlation * (1 - s2)) / s2;
    measureCollection.add("ensembleStrength", strength);
    measureCollection.add("ensembleCorrelation", correlation);
    measureCollection.add("ensembleQuality", correlation / s2);
    measureCollection.add("ensembleErrorBound", errorBound);
  }

  private static void computeMeanSquareError(EvaluationContext<? extends Ensemble> ctx) {
    Ensemble ensemble = ctx.getPredictor();
    DataFrame x = ctx.getPartition().getValidationData();
    Vector y = ctx.getPartition().getValidationTarget();
    Vector classes = ensemble.getClasses();

    DoubleAdder meanVariance = new DoubleAdder();
    DoubleAdder meanSquareError = new DoubleAdder();
    DoubleAdder meanBias = new DoubleAdder();
    DoubleAdder baseAccuracy = new DoubleAdder();
    IntStream.range(0, x.rows()).parallel().forEach(i -> {
      Vector record = x.loc().getRecord(i);
      DoubleArray c = createTrueClassVector(y, classes, i);


      /* Stores the probability of the m:th member for the j:th class */
      List<Classifier> members = ensemble.getEnsembleMembers();
      int estimators = members.size();
      DoubleArray memberEstimates = DoubleArray.zeros(estimators, classes.size());
      for (int j = 0; j < estimators; j++) {
        Classifier member = members.get(j);
        memberEstimates.setRow(j, member.estimate(record));
      }

      /* Get the mean probability vector for the i:th example */
      DoubleArray meanEstimate = Arrays.mean(0, memberEstimates);
      double variance = 0, mse = 0, bias = 0, accuracy = 0;
      for (int j = 0; j < memberEstimates.rows(); j++) {
        DoubleArray r = memberEstimates.getRow(j);
        double meanDiff = 0;
        double trueDiff = 0;
        double meanTrueDiff = 0;
        for (int k = 0; k < r.size(); k++) {
          meanDiff += Math.pow(r.get(k) - meanEstimate.get(k), 2);
          trueDiff += Math.pow(r.get(k) - c.get(k), 2);
          meanTrueDiff += Math.pow(meanEstimate.get(k) - c.get(k), 2);
        }
        variance += meanDiff;
        mse += trueDiff;
        bias += meanTrueDiff;
        accuracy += argmax(r) == find(classes, y, i) ? 1 : 0;
      }
      meanVariance.add(variance / estimators);
      meanSquareError.add(mse / estimators);
      baseAccuracy.add(accuracy / estimators);
      meanBias.add(bias / estimators);
    });

    double avgVariance = meanVariance.doubleValue() / x.rows();
    double avgBias = meanBias.doubleValue() / x.rows();
    double avgMse = meanSquareError.doubleValue() / x.rows();
    double avgBaseAccuracy = baseAccuracy.doubleValue() / x.rows();
    MeasureCollection measureCollection = ctx.getMeasureCollection();
    measureCollection.add("ensembleVariance", avgVariance);
    measureCollection.add("ensembleBias", avgBias);
    measureCollection.add("ensembleMse", avgMse);
    measureCollection.add("baseModelError", 1 - avgBaseAccuracy);
  }

  private static DoubleArray createTrueClassVector(Vector y, Vector classes, int i) {
    DoubleArray c = DoubleArray.zeros(classes.size());
    for (int j = 0; j < classes.size(); j++) {
      if (classes.loc().equals(j, y, i)) {
        c.set(j, 1);
      }
    }
    return c;
  }

  private static int argmaxnot(DoubleArray m, int not) {
    double max = Double.NEGATIVE_INFINITY;
    int argMax = -1;
    for (int i = 0; i < m.size(); i++) {
      if (not != i && m.get(i) > max) {
        argMax = i;
        max = m.get(i);
      }
    }
    return argMax;
  }

  private static double maxnot(DoubleArray m, int not) {
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < m.size(); i++) {
      if (not != i && m.get(i) > max) {
        max = m.get(i);
      }
    }
    return max;
  }

  @Override
  public void accept(EvaluationContext<? extends Ensemble> ctx) {
    computeMeanSquareError(ctx);
    computeOOBCorrelation(ctx);
  }

  @Override
  public String toString() {
    return "EnsembleEvaluator";
  }
}
