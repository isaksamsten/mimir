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

import java.util.Collection;

import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;

/**
 * The partitioner represents a strategy of how to partition a {@code DataFrame} and {@code Vector}
 * into training and validation partitions.
 *
 * <p>
 * The partitioner guarantees that the column-index of the {@code DataFrame} partitions are the same
 * as the input-{@code DataFrame}. The record-indexing might, however, be lost.
 *
 * @author Isak Karlsson
 */
public interface Partitioner<In, Out> {

  /**
   * Partitions {@code x} and {@code y} into training and validation partitions
   *
   * @param x the data
   * @param y the target
   * @return an iterable representing over the partitions
   */
  Collection<Partition<In, Out>> partition(Input<? extends In> x, Output<? extends Out> y);
}
