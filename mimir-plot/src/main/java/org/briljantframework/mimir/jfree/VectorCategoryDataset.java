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

import java.util.Collections;
import java.util.List;

import org.briljantframework.data.Is;
import org.briljantframework.data.vector.Vector;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.AbstractDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class VectorCategoryDataset extends AbstractDataset implements CategoryDataset {

  private final Vector vector;

  public VectorCategoryDataset(Vector vector) {
    this.vector = vector;
  }

  @Override
  public Comparable getRowKey(int row) {
    return (Comparable) vector.getIndex().get(row);
  }

  @Override
  public int getRowIndex(Comparable key) {
    return vector.getIndex().getLocation(key);
  }

  @Override
  public List getRowKeys() {
    return vector.getIndex();
  }

  @Override
  public Comparable getColumnKey(int column) {
    return 0;
  }

  @Override
  public int getColumnIndex(Comparable key) {
    return 0;
  }

  @Override
  public List getColumnKeys() {
    return Collections.singletonList(0);
  }

  @Override
  public Number getValue(Comparable rowKey, Comparable columnKey) {
    return vector.get(Number.class, rowKey);
  }

  @Override
  public int getRowCount() {
    return vector.size();
  }

  @Override
  public int getColumnCount() {
    return 1;
  }

  @Override
  public Number getValue(int row, int column) {
    Number value = vector.get(Number.class, vector.getIndex().get(row));
    if (Is.NA(value)) {
      return null;
    }
    return value;
  }

}
