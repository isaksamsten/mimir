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
package org.briljantframework.mimir.classification.tune;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class GridSearch<In, Out, P extends Predictor<In, Out>, O extends Predictor.Configurator<In, Out, ? extends Predictor.Learner<In, Out, ? extends P>>>
    implements Tuner<In, Out, P, O> {

  private final List<UpdatableParameter<O>> updatables = new ArrayList<>();
  private final List<String> parameterNames = new ArrayList<>();

  private Validator<In, Out, P> validator;

  public GridSearch(Validator<In, Out, P> validator) {
    this.validator = Objects.requireNonNull(validator, "validator required");
  }

  private List<Configuration<In, Out, P>> gridSearch(O builder, Input<? extends In> x,
      Output<? extends Out> y) {
    List<Configuration<In, Out, P>> configurations = new ArrayList<>();
    gridSearch(builder, x, y, configurations, new Object[updatables.size()], 0);
    return configurations;
  }

  private void gridSearch(O classifierBuilder, Input<? extends In> x, Output<? extends Out> y,
      List<Configuration<In, Out, P>> results, Object[] parameters, int n) {
    if (n != updatables.size()) {
      ParameterUpdator<O> updater = updatables.get(n).updator();
      while (updater.hasUpdate()) {
        Object value = updater.update(classifierBuilder);
        parameters[n] = value;
        gridSearch(classifierBuilder, x, y, results, parameters, n + 1);
      }
    } else {
      Predictor.Learner<In, Out, ? extends P> classifier = classifierBuilder.configure();
      Result<Out> result = validator.test(classifier, x, y);
      System.out.println(Arrays.toString(parameters));
      Vector.Builder builder = Vector.Builder.of(Object.class);
      for (int i = 0; i < parameters.length; i++) {
        builder.set(parameterNames.get(i), parameters[i]);
      }
      results.add(new Configuration<>(classifier, result, builder.build()));
    }
  }

  @Override
  public List<Configuration<In, Out, P>> tune(O toOptimize, Input<? extends In> x,
      Output<? extends Out> y) {
    return gridSearch(toOptimize, x, y);
  }


  @Override
  public Tuner<In, Out, P, O> setParameter(String name, UpdatableParameter<O> updatable) {
    Objects.requireNonNull(name, "parameter name is required");
    Objects.requireNonNull(updatable, "updatable is required");
    int i = parameterNames.indexOf(name);
    if (i < 0) {
      updatables.add(updatable);
      parameterNames.add(name);
    } else {
      updatables.set(i, updatable);
      parameterNames.set(i, name);
    }
    return this;
  }

  @Override
  public Tuner<In, Out, P, O> setValidator(Validator<In, Out, P> validator) {
    this.validator = Objects.requireNonNull(validator, "validator required");
    return this;
  }

  @Override
  public Validator<In, Out, P> getValidator() {
    return validator;
  }
}
