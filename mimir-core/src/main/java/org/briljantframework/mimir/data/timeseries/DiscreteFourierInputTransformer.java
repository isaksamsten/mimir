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
package org.briljantframework.mimir.data.timeseries;

import org.briljantframework.ComplexSequence;
import org.briljantframework.DoubleSequence;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.transform.InvertibleInputTransformer;

/**
 * Performs a discrete fourier transformation.
 * 
 * @author Isak Karlsson
 */
public class DiscreteFourierInputTransformer<T extends DoubleSequence, E extends ComplexSequence>
    implements InvertibleInputTransformer<T, E> {

  @Override
  public Input<E> transform(Input< T> x) {
    // DataSeriesCollection.Builder builder = new DataSeriesCollection.Builder(Complex.class);
    // for (Vector row : x.getRecords()) {
    // DoubleArray timeDomain = Vectors.toDoubleArray(row);
    // ComplexArray frequencyDomain = fft(timeDomain);
    // Vector.Builder rowBuilder = Type.of(Complex.class).newBuilder(timeDomain.size());
    // for (int i = 0; i < frequencyDomain.size(); i++) {
    // rowBuilder.loc().set(i, frequencyDomain.get(i));
    // }
    // builder.addRecord(rowBuilder);
    // }
    // return builder.build();
    return null;
  }

  @Override
  public E transformElement(T x) {
    return null;
  }

  @Override
  public Input<T> inverseTransform(Input< E> x) {
    // DataSeriesCollection.Builder builder = new DataSeriesCollection.Builder(Double.class);
    // for (Vector row : x.getRecords()) {
    // ComplexArray timeDomain = Vectors.toComplexArray(row);
    // DoubleArray frequencyDomain = ifft(timeDomain).asDouble();
    // Vector.Builder rowBuilder = new DoubleVector.Builder();
    // for (int i = 0; i < frequencyDomain.size(); i++) {
    // rowBuilder.loc().set(i, frequencyDomain.get(i));
    // }
    // builder.addRecord(rowBuilder);
    // }
    // return builder.build();
    return null;
  }

  @Override
  public T inverseTransform(E x) {
    return null;
  }
}
