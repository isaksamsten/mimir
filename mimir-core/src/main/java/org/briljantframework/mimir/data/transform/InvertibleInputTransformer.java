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

import org.briljantframework.mimir.data.Input;

/**
 * Some transformations are (semi) invertible, e.g. PCA. Given the transformation {@code f(x)} and
 * the inverse {@code f'(x)}, {@code f'(f(x)) ~= x}.
 * 
 * <pre>
 * InvertibleInputTransformer<String, Integer> t = new StrIntTransformer();
 * Integer i = t.transform("1");
 * assert "1".equals(t.inverseTransform(i));
 * </pre>
 * 
 * @author Isak Karlsson
 */
public interface InvertibleInputTransformer<T, E> extends InputTransformer<T, E> {

  /**
   * Inverse the transformation
   *
   * @param x an input
   * @return the inverse
   */
  Input<T> inverseTransform(Input<E> x);

  T inverseTransform(E x);
}
