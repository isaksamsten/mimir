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

import org.briljantframework.mimir.data.ArrayInput;
import org.briljantframework.mimir.data.Input;


public interface InputTransformer<T, E> extends Transformer<Input<T>, Input<E>> {

  /**
   * Transform an input
   *
   * @param x data frame to transform
   * @return transformation of input
   */
  default Input<E> transform(Input<T> x) {
    return x.stream().map(this::transformElement)
        .collect(Collectors.toCollection(() -> new ArrayInput<>(x.getProperties())));
  }

  /**
   * Transform the given input element
   * 
   * @param x the given input element
   * @return the given input element
   */
  E transformElement(T x);
}