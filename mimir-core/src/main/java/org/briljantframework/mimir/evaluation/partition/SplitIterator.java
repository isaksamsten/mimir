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

import java.util.*;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.IntArray;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class SplitIterator<In, Out> implements Iterator<Partition<In, Out>> {

  private boolean has = true;
  private final Input<In> x;
  private final List<Out> y;
  private final double splitFraction;
  private final IntArray order;

  public SplitIterator(Input<In> x, List<Out> y, double splitFraction) {
    this(x, y, splitFraction, false);
  }

  public SplitIterator(Input<In> x, List<Out> y, double splitFraction, boolean randomize) {
    Check.inRange(splitFraction, 0, 1);
    this.splitFraction = splitFraction;
    this.x = x;
    this.y = y;
    if (randomize) {
      order = Arrays.shuffle(Arrays.range(x.size()));
    } else {
      order = Arrays.range(x.size());
    }
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

    Input<In> xTraining = x.getSchema().newInput();
    List<Out> yTraining = new ArrayList<>();

    for (int i = 0; i < trainingSize; i++) {
      int ind = order.get(i);
      xTraining.add(x.get(ind));
      yTraining.add(y.get(ind));
    }

    Input<In> xValidation = x.getSchema().newInput();
    List<Out> yValidation = new ArrayList<>();
    for (int i = trainingSize; i < x.size(); i++) {
      int ind = order.get(i);
      xValidation.add(x.get(ind));
      yValidation.add(y.get(ind));
    }

    return new Partition<>(Inputs.unmodifiableInput(xTraining),
        Inputs.unmodifiableInput(xValidation), Collections.unmodifiableList(yTraining),
        Collections.unmodifiableList(yValidation));
  }
}
