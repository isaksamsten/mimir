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
package org.briljantframework.mimir.data.transform;

import static org.briljantframework.mimir.data.transform.Transformations.toDataset;

import org.briljantframework.data.SortOrder;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.series.Series;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.DecisionTree;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.classification.conformal.BootstrapConformalClassifier;
import org.briljantframework.mimir.classification.conformal.ConformalClassifier;
import org.briljantframework.mimir.classification.conformal.evaluation.ConformalClassifierValidator;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Instance;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.Outputs;
import org.briljantframework.mimir.evaluation.Result;
import org.junit.Test;

/**
 * Created by isak on 3/29/16.
 */
public class NormalizerTest {

  @Test
  public void fit() throws Exception {
    DataFrame x = DataFrames.permute(Datasets.loadIris());
    Input<Instance> in = toDataset().fitTransform(x.drop("Class"));
    Output<String> out = Outputs.asOutput(String.class, x.get("Class"));

    RandomForest.Learner<String> rf = new RandomForest.Learner<>(100);
    rf.set(DecisionTree.MAX_DEPTH, 100);

    BootstrapConformalClassifier.Learner<Instance, String> bcl =
        new BootstrapConformalClassifier.Learner<>(rf);
    bcl.set(ConformalClassifier.STOCHASTIC_SMOOTHING, false);

    ConformalClassifierValidator<Instance, String, BootstrapConformalClassifier<Instance, String>> v =
        ConformalClassifierValidator.crossValidator(10);
    Result<?> test = v.test(bcl, in, out);
    System.out.println(
        test.getMeasures().groupBy("significance", Double.class, i -> String.format("%.2f", i))
            .collect(Series::mean).sort(SortOrder.ASC));



    // LogisticRegression.Learner rf = new LogisticRegression.Learner(0.01);
    // ClassifierValidator<Instance, RandomForest> validator =
    // ClassifierValidator.crossValidator(10);
    // validator.add(EnsembleEvaluator.getInstance());
    //
    // DataFrame measures = validator.test(rf, in, out).getMeasures();
    // System.out.println(measures.reduce(Series::mean));

    // System.out.println(ClassifierValidator.crossValidate(10, rf, in,
    // out).getMeasures().reduce(Series::mean));

    // Instance a = Instance.of(0, 0);
    // Instance b = Instance.of(1, 1);
    //
    // Input<Instance> in = new ArrayInput<>();
    // in.add(a);
    // in.add(b);
    //
    // in.getProperties().set(Dataset.FEATURE_SIZE, 2);
    // in.getProperties().set(Dataset.FEATURE_NAMES, Arrays.asList("a", "b"));
    // in.getProperties().set(Dataset.FEATURE_TYPES, Arrays.asList(Double.class, Double.class));
    //
    // Output<Object> out = new ArrayOutput<>();
    // out.add(0);
    // out.add(1);
    //
    // DecisionTree.Learner dt = new DecisionTree.Learner();
    //
    // DecisionTree fit = dt.fit(in, out);
    //
    //
    // System.out.println(fit.predict(Instance.of(2, 2)));

    // ProbabilityEstimateNonconformity.Learner<Instance> penl =
    // new ProbabilityEstimateNonconformity.Learner<>(rf, ProbabilityCostFunction.margin());
    // BootstrapConformalClassifier.Learner<Instance> bccl =
    // new BootstrapConformalClassifier.Learner<>(penl);
    // ConformalClassifierValidator<Instance, BootstrapConformalClassifier<Instance>> v =
    // ConformalClassifierValidator.crossValidator(10, DoubleArray.of(0.05, 0.1, 0.2));
    // Result<?> result = v.test(bccl, in, out);
    //
    //
    // System.out.println(result.getMeasures().groupBy("significance").collect(Series::mean));
    // ProbabilityEstimateNonconformity.Learner<Instance, Object> penl =
    // new ProbabilityEstimateNonconformity.Learner<>(rf, margin());
    //
    // BootstrapConformalClassifier.Learner<Instance, Object> bccl =
    // new BootstrapConformalClassifier.Learner<>(penl);
    // Result<Object> res = ConformalClassifierValidator.crossValidator(10).test(bccl, in,
    // Outputs.asOutput(x.get("Class")));


    // System.out.println(res.getMeasures().groupBy("significance").collect(Series::mean));

  }
}
