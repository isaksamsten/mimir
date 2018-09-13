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
package org.briljantframework.mimir.classification.conformal;

import java.io.FileReader;
import java.util.List;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.parser.CsvParser;
import org.briljantframework.data.parser.Parser;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.DecisionTree;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.classification.tree.RandomSplitter;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.supervised.data.DataFrameInput;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityCostFunctionTest {

  @Test
  public void testMargin() throws Exception {
    Parser parser = new CsvParser(new FileReader(
        "/Users/isak/anaconda3/pkgs/blaze-0.9.1-py35_0/lib/python3.5/site-packages/blaze/examples/data/iris.csv"));

    DataFrame data = parser.parse();

    System.out.println(data);
    Input<Instance> input = new DataFrameInput(data.drop("species"));
    List<Object> output = data.get("species").values();

    RandomForest.Learner<Object> model = new RandomForest.Learner<>(100);
    model.set(DecisionTree.SPLITTER, RandomSplitter.sqrt());
    System.out.println(input.getSchema());

    ClassifierValidator<Instance, Object> validator = ClassifierValidator.crossValidator(10);
    Result<Object> test = validator.test(model, input, output);
    System.out.println(test.getMeasures().reduce(Series::mean));



    // InductiveConformalClassifier.Learner<Instance, Object> iccl =
    // new InductiveConformalClassifier.Learner<>(model);
    // InductiveConformalClassifier<Instance, Object> icc = iccl.fit(input, output);
    // icc.calibrate(input, output);
    // System.out.println(icc.predict(input.get(33), 0.01));



  }

  @Test
  public void testInverseProbability() throws Exception {
    MultidimensionalSchema schema = new MultidimensionalSchema(3, 0);
    schema.setAttributeName(0, "age");
    schema.setAttributeName(1, "income");
    schema.setAttributeName(2, "bmi");

    System.out.println(schema);
    Input<Instance> in = schema.newInput();
    in.add(schema.newInstance().set("age", 10).set("income", 0).set("bmi", 33).build());

    System.out.println(in);
  }
}
