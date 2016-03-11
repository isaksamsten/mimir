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

import java.util.Random;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Convert;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.VectorType;
import org.briljantframework.mimir.Input;

/**
 * Created by Isak Karlsson on 10/09/14.
 */
public abstract class AbstractSplitter implements Splitter {

  public static final int MISSING = 0;
  public static final int LEFT = -1;
  public static final int RIGHT = 1;
  /**
   * The Random.
   */
  protected final Random random = new Random();

  /**
   * Basic implementation of the splitting procedure
   *
   *
   * @param in
   * @param classSet the examples
   * @param axis the axis
   * @param threshold the threshold
   * @return the examples . split
   */
  protected TreeSplit<ValueThreshold> split(Input<? extends Vector> in, ClassSet classSet, int axis,
      Object threshold) {
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
      for (Example example : sample) {
        int direction = MISSING;
        Vector record = in.get(example.getIndex());
        if (!record.loc().isNA(axis)) {
          if (nominal) {
            direction = Is.equal(threshold, record.loc().get(axis)) ? LEFT : RIGHT;
//                axisVector.loc().get(Object.class, index).equals(threshold) ? LEFT : RIGHT;
          } else {
            double leftValue = record.loc().getAsDouble(axis);
            direction = Double.compare(leftValue, Convert.to(Double.class, threshold)) <= 0 ? LEFT : RIGHT;
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
          default:
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
      if (random.nextDouble() > 0.5) {
        left.add(example);
      } else {
        right.add(example);
      }
    }
  }


}
