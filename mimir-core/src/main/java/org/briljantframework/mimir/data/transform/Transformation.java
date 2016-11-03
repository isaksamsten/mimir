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

/**
 * Fit a InputTransformation to a dataset and return a transformation which can be used to transform
 * other datasets using the parameters of the fitted dataset. This can be particularly useful when a
 * transformation must be fitted on a dataset and applied on another. For example, in the case of
 * normalizing training and testing data.
 *
 * <p/>
 * The input
 *
 * @author Isak Karlsson
 */
public interface Transformation<T, R> {

  default <V> Transformation<T, V> followedBy(Transformer<? super R, ? extends V> after) {
    return stageOneInput -> {
      Transformer<T, R> stageOne = Transformation.this.fit(stageOneInput);
      return (Transformer<T, V>) stageTwoInput -> after.transform(stageOne.transform(stageTwoInput));
    };
  }

  default <V> Transformation<T, V> followedBy(Transformation<? super R, ? extends V> after) {
    return stageOneInput -> {
      Transformer<T, R> stageOne = Transformation.this.fit(stageOneInput);
      Transformer<? super R, ? extends V> stageTwo = after.fit(stageOne.transform(stageOneInput));
      return (Transformer<T, V>) stageTwoInput -> stageTwo.transform(stageOne.transform(stageTwoInput));
    };
  }

  default R fitTransform(T t) {
    return fit(t).transform(t);
  }

  Transformer<T, R> fit(T t);

}
