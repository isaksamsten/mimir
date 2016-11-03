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

import static org.briljantframework.mimir.classification.tree.AbstractSplitter.Direction.*;

import java.util.concurrent.ThreadLocalRandom;

import org.briljantframework.data.Is;
import org.briljantframework.data.series.Convert;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Instance;

/**
 * Created by Isak Karlsson on 10/09/14.
 */
public abstract class AbstractSplitter implements Splitter {

  protected enum Direction {
    MISSING, LEFT, RIGHT
  }

  protected TreeSplit<ValueThreshold> split(Input<? extends Instance> in, ClassSet classSet,
      int axis, Object threshold) {
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
      boolean nominal = Is.nominal(threshold);
      double numericThreshold = Double.NaN;
      if (!nominal) {
        numericThreshold = Convert.to(Double.class, threshold);
      }
      for (Example example : sample) {
        Direction direction = MISSING;
        Instance record = in.get(example.getIndex());
        Object axisValue = record.get(axis);
        if (!Is.NA(axisValue)) {
          if (nominal) {
            direction = Is.equal(threshold, axisValue) ? LEFT : RIGHT;
          } else {
            // getDouble(axis) can be optimized in some cases
            direction =
                Double.compare(record.getDouble(axis), numericThreshold) <= 0 ? LEFT : RIGHT;
          }
        }
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

    return new TreeSplit<>(left, right, ValueThreshold.create(axis, threshold));
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
