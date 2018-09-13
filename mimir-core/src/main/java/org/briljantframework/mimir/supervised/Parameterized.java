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

import org.briljantframework.mimir.Properties;
import org.briljantframework.mimir.Property;

/**
 * Created by isak on 2017-01-27.
 */
public class Parameterized {

  protected final Properties parameters;

  protected Parameterized(Properties parameters) {
    this.parameters = new Properties(parameters);
  }

  protected Parameterized() {
    this.parameters = new Properties();
  }

  public final <T> void set(Property<T> key, T value) {
    parameters.set(key, value);
  }

  public final <T> T get(Property<T> key) {
    return parameters.get(key);
  }

  protected final <T> T getOrDefault(Property<T> property, T defaultValue) {
    return parameters.getOrDefault(property, defaultValue);
  }

  protected final <T> T getOrDefault(Property<T> property) {
    return parameters.getOrDefault(property);
  }

  public final Properties getParameters() {
    return Properties.unmodifiableProperties(parameters);
  }
}
