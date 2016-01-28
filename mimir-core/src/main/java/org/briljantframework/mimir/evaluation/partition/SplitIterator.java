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

import static org.briljantframework.data.vector.Vectors.transferableBuilder;

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.briljantframework.Check;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class SplitIterator implements Iterator<Partition> {

  private boolean has = true;
  private final DataFrame x;
  private final Vector y;
  private final double splitFraction;

  public SplitIterator(DataFrame x, Vector y, double splitFraction) {
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
  public Partition next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    has = false;
    int trainingSize = x.rows() - (int) Math.round(x.rows() * splitFraction);

    DataFrame.Builder xTrainingBuilder = x.newBuilder();
    Vector.Builder yTrainingBuilder = y.newBuilder();
    for (int i = 0; i < trainingSize; i++) {
      xTrainingBuilder.addRecord(transferableBuilder(x.loc().getRecord(i)));
      yTrainingBuilder.add(y, i);
    }

    DataFrame.Builder xValidationBuilder = x.newBuilder();
    Vector.Builder yValidationBuilder = y.newBuilder();
    for (int i = trainingSize; i < x.rows(); i++) {
      xValidationBuilder.addRecord(transferableBuilder(x.loc().getRecord(i)));
      yValidationBuilder.add(y, i);
    }
    DataFrame trainingSet = xTrainingBuilder.build();
    trainingSet.setColumnIndex(x.getColumnIndex());
    DataFrame validationSet = xValidationBuilder.build();
    validationSet.setColumnIndex(x.getColumnIndex());
    return new Partition(trainingSet, validationSet, yTrainingBuilder.build(),
        yValidationBuilder.build());
  }
}
