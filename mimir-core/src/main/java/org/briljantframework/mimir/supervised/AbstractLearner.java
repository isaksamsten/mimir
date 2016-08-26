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
package org.briljantframework.mimir.supervised;

import org.briljantframework.mimir.data.Properties;
import org.briljantframework.mimir.data.Property;

/**
 * Created by isak on 3/29/16.
 */
public abstract class AbstractLearner<In, Out, P extends Predictor<In, Out>>
    implements Predictor.Learner<In, Out, P> {

  private final Properties parameters;

  protected AbstractLearner(Properties parameters) {
    this.parameters = new Properties(parameters);
  }

  protected AbstractLearner() {
    this.parameters = new Properties();
  }

  @Override
  public final <T> void set(Property<T> key, T value) {
    parameters.set(key, value);
  }

  @Override
  public final <T> T get(Property<T> key) {
    return parameters.get(key);
  }

  protected <T> T getOrDefault(Property<T> property, T defaultValue) {
    return parameters.getOrDefault(property, defaultValue);
  }

  protected <T> T getOrDefault(Property<T> property) {
    return parameters.getOrDefault(property);
  }

  @Override
  public final Properties getParameters() {
    return parameters;
  }

  @Override
  public String toString() {
    return getClass().getSimpleName() + "{ parameters: " + getParameters() + "}";
  }
}
