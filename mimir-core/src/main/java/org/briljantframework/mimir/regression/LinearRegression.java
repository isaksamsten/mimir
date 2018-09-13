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
package org.briljantframework.mimir.regression;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.Property;
import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.supervised.*;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class LinearRegression implements Regression<Instance> {

  public static final Property<Double> REGULARIZATION =
      Property.of("regularization", Double.class, 1.0);

  private final DoubleArray theta;
  private final MultidimensionalSchema schema;

  private LinearRegression(MultidimensionalSchema schema, DoubleArray theta) {
    this.schema = schema;
    this.theta = theta;
  }

  public DoubleArray getParameters() {
    return theta.copy();
  }

  @Override
  public Double predict(Instance instance) {
    Check.argument(schema.isValid(instance), "illegal instance");
    DoubleArray x = Arrays
        .concatenate(java.util.Arrays.asList(DoubleArray.of(1), instance.getNumericalAttributes()));
    return Arrays.inner(theta, x);
  }

  @Override
  public List<Double> predict(Input<Instance> x) {
    List<Double> output = new ArrayList<>();
    for (Instance instance : x) {
      output.add(predict(instance));
    }
    return Collections.unmodifiableList(output);
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner extends Predictor.Learner<Instance, Double, LinearRegression> {

    public Learner() {}

    @Override
    public LinearRegression fit(Input<Instance> in, List<Double> out) {
      Check.argument(in.getSchema() instanceof MultidimensionalSchema);

      MultidimensionalSchema schema = (MultidimensionalSchema) in.getSchema();
      Check.argument(!schema.hasCategoricalAttributes(), "only numerical attributes supported");

      DoubleArray x = Arrays.hstack(Arrays.ones(in.size(), 1), schema.getNumericalAttributes(in));
      DoubleArray y = DoubleArray.copyOf(out).reshape(out.size(), 1);

      double lambda = getOrDefault(REGULARIZATION);
      DoubleArray regularization = Arrays.times(Arrays.eye(x.columns()), lambda);

      DoubleArray inner = Arrays.dot(Arrays.dot(x.transpose(), x), regularization);
      DoubleArray inverse = Arrays.dot(Arrays.linalg.pinv(inner), x.transpose());
      DoubleArray theta = Arrays.dot(inverse, y);
      return new LinearRegression(schema, theta);
    }

  }
}
