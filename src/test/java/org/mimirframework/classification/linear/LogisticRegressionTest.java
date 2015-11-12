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

package org.mimirframework.classification.linear;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.RandomForest;
import org.briljantframework.data.Collectors;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.mimirframework.evaluation.Result;
import org.junit.Test;

public class LogisticRegressionTest {

  @Test
  public void testLogisticRegression() throws Exception {
    DataFrame data = data(100, 10);

    // DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = data.drop("Class");
    // x = kernel(x);

    Vector y = data.get("Class");// .satisfies(String.class, v -> v.equals("Iris-setosa"));
    Classifier.Learner<? extends Classifier> classifier = new RandomForest.Learner(100);
    // new LogisticRegression.Configurator(500).setRegularization(1).configure();
    // LogisticRegression model = (LogisticRegression) classifier.fit(x, y);

    // System.out.println(model.getOddsRatio("(Intercept)"));
    // for (Object o : x.getColumnIndex().keySet()) {
    // System.out.println(model.getOddsRatio(o));
    // }

    System.out.println(classifier);
    long start = System.nanoTime();
    Result result = ClassifierValidator.crossValidation(10).test((d, t) -> {
      return classifier.fit(kernel(d), t);
    } , x, y);
    System.out.println((System.nanoTime() - start) / 1e6);
    System.out.println(result.getMeasures().mean());
  }

  private DataFrame data(int m, int c) {
    DataFrame.Builder builder = DataFrame.builder();
    NormalDistribution distribution = new NormalDistribution();
    for (int i = 0; i < m / 2; i++) {
      builder.addRecord(Vectors.transferableBuilder(Vector.fromSupplier(distribution::sample, 2)));
    }

    for (int i = 0; i < m / 4; i++) {
      builder.addRecord(Vectors.transferableBuilder(
          Vector.fromSupplier(distribution::sample, 2).sub(Vector.of(1.5, 1.5))));
    }

    for (int i = 0; i < m / 4; i++) {
      builder.addRecord(Vectors.transferableBuilder(
          Vector.fromSupplier(distribution::sample, 2).add(Vector.of(1.5, 1.5))));
    }
    builder.set("Class", Vector.of(0, 1).collect(Collectors.each(100 / 2)));

    return DataFrames.permuteRecords(builder.build());
  }

  private DataFrame kernel(DataFrame x) {
    DoubleArray y = x.toDoubleArray();

    DoubleArray result = Arrays.newDoubleArray(x.rows(), x.rows());
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < x.rows(); j++) {
        // result.set(i, j, Math.pow(1 * Arrays.dot(y.getRow(i), y.getRow(j)) + 1, 10));
        result.set(i, j, -0.000001 * Math.exp(Arrays.norm2(y.getRow(i).sub(y.getRow(j)))));
      }
    }

    DataFrame.Builder builder = x.newBuilder();
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < x.columns(); j++) {
        builder.loc().set(i, j, y.get(i, j));
      }
    }
    return builder.build();
  }

  @Test
  public void testOdds() throws Exception {
    DataFrame x = DataFrame.of("Age", Vector.of(55, 28, 65, 46, 86, 56, 85, 33, 21, 42), "Smoker",
        Vector.of(0, 0, 1, 0, 1, 1, 0, 0, 1, 1));
    Vector y = Vector.of(0, 0, 0, 1, 1, 1, 0, 0, 0, 1);
    System.out.println(x);

    org.mimirframework.classification.LogisticRegression.Learner regression = new org.mimirframework.classification.LogisticRegression.Learner();
    org.mimirframework.classification.LogisticRegression model = regression.fit(x, y);
    System.out.println(model);

    System.out.println("(Intercept) " + model.getOddsRatio("(Intercept)"));
    for (Object o : x.getColumnIndex().keySet()) {
      System.out.println(o + " " + model.getOddsRatio(o));
    }
  }
}
