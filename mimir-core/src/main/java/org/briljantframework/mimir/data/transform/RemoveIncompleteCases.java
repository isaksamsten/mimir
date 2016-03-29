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


import java.util.stream.Collectors;

import org.briljantframework.data.Is;
import org.briljantframework.mimir.data.ArrayInput;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;
import org.briljantframework.mimir.data.Instance;

/**
 * Removes incomplete cases, i.e. rows with missing values.
 *
 * @author Isak Karlsson
 */
public class RemoveIncompleteCases<T extends Instance> implements Transformer<T, T> {

  @Override
  public Input<T> transform(Input<? extends T> x) {
    Input<T> o = x.stream().filter(instance -> !hasNA(instance))
        .collect(Collectors.toCollection(() -> new ArrayInput<>(x.getProperties())));
    return Inputs.unmodifiableInput(o);
  }

  private boolean hasNA(T instance) {
    for (int i = 0; i < instance.size(); i++) {
      if (Is.NA(instance.get(i))) {
        return true;
      }
    }
    return false;
  }

  @Override
  public T transform(T x) {
    if (!hasNA(x))
      return x;
    else {
      throw new IllegalStateException("no instance returned");
    }
  }
}
