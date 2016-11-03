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
package org.briljantframework.mimir.jfree;

import java.util.List;

import org.briljantframework.data.series.Series;
import org.jfree.data.general.AbstractDataset;
import org.jfree.data.general.PieDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class SeriesPieDataset extends AbstractDataset implements PieDataset {

  private final Series vector;

  public SeriesPieDataset(Series vector) {
    this.vector = vector;
  }

  @Override
  public Comparable getKey(int index) {
    return (Comparable) vector.index().get(index);
  }

  @Override
  public int getIndex(Comparable key) {
    return vector.index().getLocation(key);
  }

  @Override
  public List getKeys() {
    return vector.index();
  }

  @Override
  public Number getValue(Comparable key) {
    return vector.get(Number.class, key);
  }

  @Override
  public int getItemCount() {
    return vector.size();
  }

  @Override
  public Number getValue(int index) {
    return vector.get(Number.class, vector.index().get(index));
  }
}
