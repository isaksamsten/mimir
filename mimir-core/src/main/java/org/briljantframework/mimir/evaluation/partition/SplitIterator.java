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
package org.briljantframework.mimir.evaluation.partition;

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.briljantframework.Check;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.ArrayInput;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.ArrayOutput;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class SplitIterator<In, Out> implements Iterator<Partition<In, Out>> {

  private boolean has = true;
  private final Input<In> x;
  private final Output<Out> y;
  private final double splitFraction;

  public SplitIterator(Input<In> x, Output<Out> y, double splitFraction) {
    Check.inRange(splitFraction, 0, 1);
    this.splitFraction = splitFraction;
    this.x = x;
    this.y = y;

  }

  @Override
  public boolean hasNext() {
    return has;
  }

  @Override
  public Partition<In, Out> next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    has = false;
    int trainingSize = x.size() - (int) Math.round(x.size() * splitFraction);

    Input<In> xTraining = new ArrayInput<>(x.getProperties());
    Output<Out> yTraining = new ArrayOutput<>();

    for (int i = 0; i < trainingSize; i++) {
      xTraining.add(x.get(i));
      yTraining.add(y.get(i));
    }

    Input<In> xValidation = new ArrayInput<>(x.getProperties());
    Output<Out> yValidation = new ArrayOutput<>();
    for (int i = trainingSize; i < x.size(); i++) {
      xValidation.add(x.get(i));
      yValidation.add(y.get(i));
    }
    // DataFrame trainingSet = xTrainingBuilder.build();
    // trainingSet.setColumnIndex(x.getColumnIndex());
    // DataFrame validationSet = xValidationBuilder.build();
    // validationSet.setColumnIndex(x.getColumnIndex());
    return new Partition<>(xTraining, xValidation, yTraining, yValidation);
  }
}
