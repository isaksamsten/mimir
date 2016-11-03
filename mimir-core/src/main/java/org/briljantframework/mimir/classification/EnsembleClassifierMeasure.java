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
package org.briljantframework.mimir.classification;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

import org.briljantframework.array.*;
import org.briljantframework.data.Is;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class EnsembleClassifierMeasure {
  private double oobError;
  private double strength;
  private double correlation;
  private double variance;
  private double bias;
  private double mse;
  private double baseModelError;

  public <In> EnsembleClassifierMeasure(Ensemble<In, ?> ensemble, Input<? extends In> trainingData,
      Output<?> trainingTarget, Input<? extends In> validationData, Output<?> validationTarget) {
    initializeStrengthCorrelation(ensemble, trainingData, trainingTarget);
    initializeBiasVarianceDecomposition(ensemble, validationData, validationTarget);
  }

  public double getOobError() {
    return oobError;
  }

  public double getStrength() {
    return strength;
  }

  public double getCorrelation() {
    return correlation;
  }

  public double getErrorBound() {
    double s2 = Math.pow(getStrength(), 2);
    return (correlation * (1 - s2)) / s2;
  }

  public double getQuality() {
    return getCorrelation() / Math.pow(getStrength(), 2);
  }

  public double getVariance() {
    return variance;
  }

  public double getBias() {
    return bias;
  }

  public double getMeanSquareError() {
    return mse;
  }

  public double getBaseModelError() {
    return baseModelError;
  }

  @Override
  public String toString() {
    return "EnsembleClassifierMeasure{" + "oobError=" + oobError + ", strength=" + strength
        + ", correlation=" + correlation + ", variance=" + variance + ", bias=" + bias + ", mse="
        + mse + ", baseModelError=" + baseModelError + '}';
  }

  private static int argmaxExcluding(DoubleArray m, int not) {
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

  private <In> void initializeStrengthCorrelation(Ensemble<In, ?> ensemble, Input<? extends In> x,
      Output<?> y) {
    Array<?> classes = ensemble.getClasses();
    BooleanArray oobIndicator = ensemble.getOobIndicator();
    List<? extends ProbabilityEstimator<In, ?>> members = ensemble.getEnsembleMembers();

    // Store the out-of-bag and in-bag probability estimates
    DoubleArray oobEstimates = DoubleArray.zeros(x.size(), classes.size());
    DoubleArray inbEstimates = DoubleArray.zeros(x.size(), classes.size());

    // Count the number of times each training sample have been included
    IntArray counts = Arrays.sum(1, oobIndicator.intArray());

    // Compute the in-bag and out-of-bag estimates for all examples
    DoubleAdder oobAccuracy = new DoubleAdder();
    IntStream.range(0, x.size()).parallel().forEach(i -> {
      int inbSize = members.size() - counts.get(i);
      int oobSize = counts.get(i);
      In record = x.get(i);
      for (int j = 0; j < members.size(); j++) {
        DoubleArray estimate = members.get(j).estimate(record);
        if (oobIndicator.get(i, j)) {
          oobEstimates.getRow(i).combineAssign(estimate, (e, v) -> e + v / oobSize);
        } else {
          inbEstimates.getRow(i).combineAssign(estimate, (e, v) -> e + v / inbSize);
        }
      }
      // Vectors.find(classes, y, i)
      oobAccuracy.add(classes.indexOf(y.get(i)) == Arrays.argmax(oobEstimates.getRow(i)) ? 1 : 0);
    });
    double avgOobAccuracy = oobAccuracy.sum() / x.size();
    this.oobError = 1 - avgOobAccuracy;

    DoubleAdder strengthA = new DoubleAdder();
    DoubleAdder strengthSquareA = new DoubleAdder();
    IntStream.range(0, oobEstimates.rows()).parallel().forEach(i -> {
      DoubleArray estimation = oobEstimates.getRow(i);
      int c = classes.indexOf(y.get(i)); // Vectors.find(classes, y, i);
      double ma = estimation.get(c) - Arrays.maxExcluding(estimation, c);
      strengthA.add(ma);
      strengthSquareA.add(ma * ma);
    });

    double strength = strengthA.doubleValue() / y.size();
    double strengthSquare = strengthSquareA.doubleValue() / y.size();
    double s2 = strength * strength;
    double variance = strengthSquare - s2;
    double std = 0;
    for (int j = 0; j < members.size(); j++) {
      ProbabilityEstimator<In,?> member = members.get(j);
      AtomicInteger oobSizeA = new AtomicInteger(0);
      DoubleAdder p1A = new DoubleAdder();
      DoubleAdder p2A = new DoubleAdder();
      final int memberIndex = j;
      IntStream.range(0, x.size()).parallel().forEach(i -> {
        if (oobIndicator.get(i, memberIndex)) {
          oobSizeA.getAndIncrement();
          int c = classes.indexOf(y.get(i)); // Vectors.find(classes, y, i);
          DoubleArray memberEstimation = member.estimate(x.get(i));
          DoubleArray ibEstimation = inbEstimates.getRow(i);
          p1A.add(Arrays.argmax(memberEstimation) == c ? 1 : 0);
          p2A.add(Arrays.argmax(memberEstimation) == argmaxExcluding(ibEstimation, c) ? 1 : 0);
        }
      });
      double p1 = p1A.sum() / oobSizeA.get();
      double p2 = p2A.sum() / oobSizeA.get();
      std += Math.sqrt(p1 + p2 + Math.pow(p1 - p2, 2));
    }
    std = Math.pow(std / members.size(), 2);
    double correlation = variance / std;
    this.strength = strength;
    this.correlation = correlation;
    // measureCollection.add("ensembleStrength", strength);
    // measureCollection.add("ensembleCorrelation", correlation);


  }

  private <In> void initializeBiasVarianceDecomposition(Ensemble<In,?> ensemble,
      Input<? extends In> x, Output<?> y) {
    Array<?> classes = ensemble.getClasses();

    DoubleAdder meanVariance = new DoubleAdder();
    DoubleAdder meanSquareError = new DoubleAdder();
    DoubleAdder meanBias = new DoubleAdder();
    DoubleAdder baseAccuracy = new DoubleAdder();
    IntStream.range(0, x.size()).parallel().forEach(i -> {
      In record = x.get(i);
      DoubleArray c = createTrueClassVector(classes, y.get(i));


      /* Stores the probability of the m:th member for the j:th class */
      List<? extends ProbabilityEstimator<In, ?>> members = ensemble.getEnsembleMembers();
      int estimators = members.size();
      DoubleArray memberEstimates = DoubleArray.zeros(estimators, classes.size());
      for (int j = 0; j < estimators; j++) {
        ProbabilityEstimator<In,?> member = members.get(j);
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
        accuracy += Arrays.argmax(r) == classes.indexOf(y.get(i)) ? 1 : 0; // Vectors.find(classes,
                                                                           // y, i)
      }
      meanVariance.add(variance / estimators);
      meanSquareError.add(mse / estimators);
      baseAccuracy.add(accuracy / estimators);
      meanBias.add(bias / estimators);
    });

    this.variance = meanVariance.doubleValue() / x.size();
    this.bias = meanBias.doubleValue() / x.size();
    this.mse = meanSquareError.doubleValue() / x.size();
    this.baseModelError = 1 - baseAccuracy.doubleValue() / x.size();
  }

  private static DoubleArray createTrueClassVector(Array<?> classes, Object label) {
    DoubleArray c = DoubleArray.zeros(classes.size());
    for (int j = 0; j < classes.size(); j++) {
      if (Is.equal(classes.get(j), label)) {
        c.set(j, 1);
      }
    }
    return c;
  }
}
