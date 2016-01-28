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
/**
 * This copy of Woodstox XML processor is licensed under the
 * Apache (Software) License, version 2.0 ("the License").
 * See the License for details about distribution rights, and the
 * specific rights regarding derivate works.
 *
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/
 *
 * A copy is also included in the downloadable source code package
 * containing Woodstox, in file "ASL2.0", under the same directory
 * as this file.
 */
package org.briljantframework.examples;

import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.classification.conformal.ClassifierCalibrator;
import org.briljantframework.mimir.classification.conformal.ClassifierNonconformity;
import org.briljantframework.mimir.classification.conformal.InductiveConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ProbabilityCostFunction;
import org.briljantframework.mimir.classification.conformal.ProbabilityEstimateNonconformity;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class InductiveConformalPredictionExample {
  public static void main(String[] args) {
    // Load the iris data set
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());

    // Remove the class variable from the input data and set each NA value to the column mean
    DataFrame x = iris.drop("Class");// .apply(v -> v.set(v.where(Is::NA), v.mean()));

    // Get the class variable
    Vector y = iris.get("Class");

    // Create a classifier learner to use for estimating the non-conformity scores
    Predictor.Learner<? extends Classifier> classifier =
        new RandomForest.Configurator(100).configure();

    // Initialize the non-conformity learner using the margin as cost function
    ClassifierNonconformity.Learner nc =
        new ProbabilityEstimateNonconformity.Learner(classifier, ProbabilityCostFunction.margin());

    // Initialize an inductive conformal classifier using the non-conformity learner
    // and calibrator
    ClassifierCalibrator calibrator = ClassifierCalibrator.unconditional();
    InductiveConformalClassifier.Learner cp =
        new InductiveConformalClassifier.Learner(nc, calibrator);

    // Creates a validator for evaluating the validity and efficiency of the conformal classifier.
    // In this case, we evaluate the classifier using 10-fold cross-validation and 10 significance
    // levels between 0.1 and 0.1
    Validator<InductiveConformalClassifier> validator =
        ConformalClassifierValidator.crossValidator(10, 0.25);

    Result result = validator.test(cp, x, y);

    // Get the measures
    DataFrame measures = result.getMeasures();

    // Compute the mean of all measures grouped by significance level
    // Only selecting the significance level, error rate and number of classes
    DataFrame meanPerSignificance =
        measures.select("significance", "error", "noClasses")
            .groupBy(Double.class, v -> String.format("%.2f", v), "significance")
            .collect(Vector::mean).sort(SortOrder.ASC);

    System.out.println(meanPerSignificance);
  }
}
