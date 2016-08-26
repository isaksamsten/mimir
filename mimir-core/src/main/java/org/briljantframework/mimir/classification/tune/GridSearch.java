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
import java.util.List;
import java.util.Objects;

import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.Properties;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class GridSearch<In, Out, P extends Predictor<In, Out>> {

  // private final List<String> parameterNames = new ArrayList<>();
  private final List<Updatable> updatables = new ArrayList<>();
  private Validator<In, Out, P> validator;

  public GridSearch(Validator<In, Out, P> validator) {
    this.validator = Objects.requireNonNull(validator, "validator required");
  }

  private List<Configuration<Out>> gridSearch(Predictor.Learner<In, Out, ? extends P> classifier,
      Input<? extends In> x, Output<? extends Out> y) {
    List<Configuration<Out>> configurations = new ArrayList<>();
    gridSearch(classifier, x, y, configurations, new Object[updatables.size()], 0);
    return configurations;
  }

  private void gridSearch(Predictor.Learner<In, Out, ? extends P> classifier, Input<? extends In> x,
      Output<? extends Out> y, List<Configuration<Out>> results, Object[] parameters, int n) {

    if (n != updatables.size()) {
      Updater updater = updatables.get(n).updator();
      while (updater.hasUpdate()) {
        Object value = updater.update(classifier);
        parameters[n] = value;
        gridSearch(classifier, x, y, results, parameters, n + 1);
      }
    } else {
      Result<Out> result = validator.test(classifier, x, y);
      results.add(new Configuration<>(result, new Properties(classifier.getParameters())));
    }
  }


  public List<Configuration<Out>> tune(Predictor.Learner<In, Out, ? extends P> toOptimize,
      Input<? extends In> x, Output<? extends Out> y) {
    return gridSearch(toOptimize, x, y);
  }

  public void add(Updatable updatable) {
    Objects.requireNonNull(updatable, "updatable is required");
    updatables.add(updatable);
  }
}
