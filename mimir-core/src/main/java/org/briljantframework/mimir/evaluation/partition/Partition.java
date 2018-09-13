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

import org.briljantframework.Check;
import org.briljantframework.mimir.data.Input;

import java.util.List;

/**
 * @author Isak Karlsson
 */
public final class Partition<In, Out> {

  private final Input<In> trainingX, validationX;
  private final List<Out> trainingY, validationY;

  public Partition(Input<In> trainingX, Input<In> validationX,
                   List<Out> trainingY, List<Out> validationY) {
    this.trainingX = trainingX;
    this.validationX = validationX;
    this.trainingY = trainingY;
    this.validationY = validationY;
  }

  /**
   * Get the data intended for training
   *
   * @return the training data
   */
  public Input<In> getTrainingData() {
    Check.state(trainingX != null, "No training data available");
    return trainingX;
  }

  /**
   * Get the target intended for training
   *
   * @return the training target
   */
  public List<Out> getTrainingTarget() {
    Check.state(trainingY != null, "No training target available");
    return trainingY;
  }

  /**
   * Get the data intended for validation
   *
   * @return the validation data
   */
  public Input<In> getValidationData() {
    Check.state(trainingY != null, "No validation data available");
    return validationX;
  }

  /**
   * Get the target intended for validation
   *
   * @return the validation target
   */
  public List<Out> getValidationTarget() {
    Check.state(trainingY != null, "No validation target available");
    return validationY;
  }
}
