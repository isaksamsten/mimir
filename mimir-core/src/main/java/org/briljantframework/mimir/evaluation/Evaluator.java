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
package org.briljantframework.mimir.evaluation;


import java.util.function.IntConsumer;

import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public interface Evaluator<P extends Predictor> {

  static Evaluator<Predictor> foldOutput(IntConsumer consumer) {
    return new Evaluator<Predictor>() {
      private int fold = 0;

      @Override
      public void accept(EvaluationContext<? extends Predictor> ctx) {
        consumer.accept(fold++);
      }
    };
  }

  /**
   * Performs a modification to the evaluation context. For example, adding a measure.
   *
   * @param ctx the evaluation context
   */
  void accept(EvaluationContext<? extends P> ctx);
}
