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
import org.briljantframework.array.Range;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;

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
 *   DataFrame df = MixedDataFrame.of(&quot;a&quot;, Vector.of(1, 2, 3, 4, 5), &quot;b&quot;, Vector.of(5, 4, 3, 2, 1),
 *       &quot;c&quot;, Vector.of(4, 3, 2, 2, 1));
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
 * {@link ClassifierValidator#crossValidator(int)}
 *
 * @author Isak Karlsson
 */
public class FoldIterator<In, Out> implements Iterator<Partition<In, Out>> {

  private final int folds, foldSize, rows;
  private final Input<In> x;
  private final List<Out> y;
  private final int reminder;

  private int current = 0;
  private IntArray order;

  public FoldIterator(Input<In> x, List<Out> y, int folds) {
    this(x, y, folds, false);
  }

  public FoldIterator(Input<In> x, List<Out> y, int folds, boolean random) {
    Check.argument(x.size() == y.size(), "Input and output must be of equal size.");
    Check.argument(folds > 1 && folds <= x.size(), "Invalid fold count.");

    this.x = Objects.requireNonNull(x);
    this.y = Objects.requireNonNull(y);
    this.rows = x.size();
    this.folds = folds;
    this.foldSize = rows / folds;
    this.reminder = rows % folds;
    if (random) {
      order = Arrays.shuffle(Range.of(x.size()));
    } else {
      order = Range.of(x.size());
    }
  }

  @Override
  public boolean hasNext() {
    return current < folds;
  }

  @Override
  public Partition<In, Out> next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }

    current += 1;
    Input<In> xTraining = x.getSchema().newInput();
    List<Out> yTraining = new ArrayList<>();

    Input<In> xValidation = x.getSchema().newInput();
    List<Out> yValidation = new ArrayList<>();

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
      int ind = order.get(index);
      xTraining.add(x.get(ind));
      yTraining.add(y.get(ind));
      index += 1;
    }

    // Part 2: this is a validation part. Add the second
    // next foldSize * current examples until validation end
    int validationEnd = foldEnd + foldSize;
    for (int i = trainingEnd; i < validationEnd; i++) {
      int ind = order.get(index);
      xValidation.add(x.get(ind));
      yValidation.add(y.get(ind));
      index += 1;
    }

    // Part 3: this is a training part
    for (int i = validationEnd; i < rows; i++) {
      int ind = order.get(index);
      xTraining.add(x.get(ind));
      yTraining.add(y.get(ind));
      index += 1;
    }

    return new Partition<>(Inputs.unmodifiableInput(xTraining),
        Inputs.unmodifiableInput(xValidation), Collections.unmodifiableList(yTraining),
        Collections.unmodifiableList(yValidation));
  }
}
