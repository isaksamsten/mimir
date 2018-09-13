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
package org.briljantframework.mimir.classification.tree;

import java.util.concurrent.ThreadLocalRandom;

import org.briljantframework.mimir.data.Input;

/**
 * Created by Isak Karlsson on 10/09/14.
 */
public abstract class AbstractSplitter<T> implements Splitter<T> {

  protected TreeSplit<T> split(Input<? extends T> in, ClassSet classSet,
      TreeNodeTest<T> tester) {
    ClassSet left = new ClassSet(classSet.getDomain());
    ClassSet right = new ClassSet(classSet.getDomain());

    /*
     * Partition every class separately
     */
    for (ClassSet.Sample sample : classSet.samples()) {
      Object target = sample.getTarget();

      ClassSet.Sample leftSample = ClassSet.Sample.create(target);
      ClassSet.Sample rightSample = ClassSet.Sample.create(target);
      ClassSet.Sample missingSample = ClassSet.Sample.create(target);

      /*
       * STEP 1: Partition the examples according to threshold
       */
      for (Example example : sample) {
        T record = in.get(example.getIndex());
        Direction direction = tester.test(record);
        switch (direction) {
          case LEFT:
            leftSample.add(example);
            break;
          case RIGHT:
            rightSample.add(example);
            break;
          case MISSING:
            missingSample.add(example);
        }
      }

      /*
       * STEP 2: Distribute examples with missing getPosteriorProbabilities
       */
      distributeMissing(leftSample, rightSample, missingSample);

      /*
       * STEP 3: Ignore classes with no examples in the partition
       */
      if (!leftSample.isEmpty()) {
        left.add(leftSample);
      }
      if (!rightSample.isEmpty()) {
        right.add(rightSample);
      }
    }

    return new TreeSplit<>(left, right, tester);
  }

  /**
   * Distribute missing getPosteriorProbabilities (this should be an injected dependency)
   *
   * @param left the left
   * @param right the right
   * @param missing the missing
   */
  protected void distributeMissing(ClassSet.Sample left, ClassSet.Sample right,
      ClassSet.Sample missing) {
    for (Example example : missing) {
      if (ThreadLocalRandom.current().nextDouble() > 0.5) {
        left.add(example);
      } else {
        right.add(example);
      }
    }
  }


}
