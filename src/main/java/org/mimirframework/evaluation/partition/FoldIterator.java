/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.evaluation.partition;

import static org.briljantframework.data.vector.Vectors.transferableBuilder;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;

import org.briljantframework.Check;
import org.mimirframework.classification.ClassifierValidator;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * Lazy iterator that partitions the supplied {@linkplain DataFrame data frame} and {@code Vector
 * target vector} into {@code n} disjoint <em>folds</em> of <em>training</em> and
 * <em>validation</em> partitions.
 *
 * <p/>
 * For example, Given the data frame {@code x}
 *
 * <pre>
 * {
 *   &#064;code
 *   DataFrame df =
 *       MixedDataFrame.of(&quot;a&quot;, Vector.of(1, 2, 3, 4, 5), &quot;b&quot;, Vector.of(5, 4, 3, 2, 1), &quot;c&quot;,
 *           Vector.of(4, 3, 2, 2, 1));
 * }
 * </pre>
 *
 * which produces:
 *
 * <pre>
 *    a  b  c
 * 0  1  5  4
 * 1  2  4  3
 * 2  3  3  2
 * 3  4  2  2
 * 4  5  1  1
 * 
 * [5 rows x 3 columns]
 * </pre>
 *
 * Then,
 * 
 * <pre>
 * {@code
 * FoldIterator iter = new FoldIterator(df.drop("a"), df.get("a"), 2)
 * Partition part = iter.next()
 * }
 * </pre>
 *
 * produces a first (and a second) <em>training</em> partition {@code part.getTrainingData()}:
 *
 * <pre>
 *    b  c
 * 0  5  4
 * 1  4  3
 * 
 * [2 rows x 2 columns]
 * </pre>
 *
 * and a first (and a second) <em>validation</em> partition {@code part.getValidationData()}:
 * 
 * <pre>
 *    b  c
 * 0  3  2
 * 1  2  2
 * 2  1  1
 * 
 * [3 rows x 2 columns]
 * </pre>
 *
 * This class can be used to implement cross-validation. For an implementation, see
 * {@link ClassifierValidator#crossValidation(int)}
 *
 * @author Isak Karlsson
 */
public class FoldIterator implements Iterator<Partition> {

  private final int folds, foldSize, rows;
  private final DataFrame x;
  private final Vector y;
  private final int reminder;

  private int current = 0;

  public FoldIterator(DataFrame x, Vector y, int folds) {
    Check.argument(x.rows() == y.size(), "Data and target must be of equal size.");
    Check.argument(folds > 1 && folds <= x.rows(), "Invalid fold count.");

    this.x = Objects.requireNonNull(x);
    this.y = Objects.requireNonNull(y);
    this.rows = x.rows();
    this.folds = folds;
    this.foldSize = rows / folds;
    this.reminder = rows % folds;
  }

  @Override
  public boolean hasNext() {
    return current < folds;
  }

  @Override
  public Partition next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }

    current += 1;
    DataFrame.Builder xTrainingBuilder = x.newBuilder();
    Vector.Builder yTrainingBuilder = y.newBuilder();

    DataFrame.Builder xValidationBuilder = x.newBuilder();
    Vector.Builder yValidationBuilder = y.newBuilder();

    int index = 0;
    int foldEnd = rows - foldSize * current;

    // Account for the case when rows % folds != 0
    // by adding an extra validation example to the
    // first `reminder` folds
    int pad = 0;
    if (current <= this.reminder) {
      pad = 1;
    }

    // Part 1: this is a training part add the first
    // foldSize * current examples as training examples
    int trainingEnd = foldEnd - pad;
    for (int i = 0; i < trainingEnd; i++) {
      xTrainingBuilder.loc().setRecord(i, transferableBuilder(x.loc().getRecord(index)));
      yTrainingBuilder.add(y, index);
      index += 1;
    }

    // Part 2: this is a validation part. Add the second
    // next foldSize * current examples until validation end
    int newIndex = 0;
    int validationEnd = foldEnd + foldSize;
    for (int i = trainingEnd; i < validationEnd; i++) {
      xValidationBuilder.loc().setRecord(newIndex, transferableBuilder(x.loc().getRecord(index)));
      yValidationBuilder.add(y, index);
      index += 1;
      newIndex += 1;
    }

    // Part 3: this is a training part
    newIndex = trainingEnd;
    for (int i = validationEnd; i < rows; i++) {
      xTrainingBuilder.loc().setRecord(newIndex, transferableBuilder(x.loc().getRecord(index)));
      yTrainingBuilder.add(y, index);
      index += 1;
      newIndex += 1;
    }

    DataFrame trainingSet = xTrainingBuilder.build();
//    trainingSet.setColumnIndex(x.getColumnIndex());
    DataFrame validationSet = xValidationBuilder.build();
//    validationSet.setColumnIndex(x.getColumnIndex());

    Vector yTraining = yTrainingBuilder.build();
    Vector yValidation = yValidationBuilder.build();
    return new Partition(trainingSet, validationSet, yTraining, yValidation);
  }
}
