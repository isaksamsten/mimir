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
package org.briljantframework.mimir.supervised.data;

import java.util.Arrays;
import java.util.List;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.MutableInput;
import org.briljantframework.mimir.data.Schema;

/**
 * Created by isak on 2017-02-24.
 */
public class MultidimensionalSchema implements Schema<Instance> {
  private static final String ILLEGAL_SCHEMA = "input has illegal schema";

  private final int numericalAttributes;
  private final int categoricalAttributes;
  private String[] attributeNames;
  private final int attributes;

  public MultidimensionalSchema(int numericalAttributes, int categoricalAttributes) {
    this.numericalAttributes = numericalAttributes;
    this.categoricalAttributes = categoricalAttributes;
    this.attributes = categoricalAttributes + numericalAttributes;
    this.attributeNames = new String[numericalAttributes + categoricalAttributes];
  }

  public InstanceBuilder newInstance() {
    return new MultidimensionalBuilder();
  }

  @Override
  public Input<Instance> newInput() {
    MultidimensionalSchema schema =
        new MultidimensionalSchema(numericalAttributes, categoricalAttributes);
    schema.attributeNames = attributeNames.clone();
    return new MutableInput<>(schema);
  }

  protected int containsAttribute(String name) {
    if (name == null) {
      return -1;
    }

    int i = 0;
    for (String attributeName : attributeNames) {
      if (name.equals(attributeName)) {
        return i;
      }
      i++;
    }

    return -1;
  }

  public void setAttributeName(int index, String name) {
    Check.index(index, attributes());
    Check.argument(containsAttribute(name) < 0, "attribute name already exists.");
    attributeNames[index] = name;
  }

  public String getAttributeName(int index) {
    String attributeName = attributeNames[index];
    return attributeName != null ? attributeName : String.format("Attribute %d", index);
  }

  public List<String> getAttributeNames() {
    return Arrays.asList(attributeNames);
  }

  /**
   * Returns the number of numerical attributes.
   *
   * @return the number of numerical attributes.
   */
  public final int numericalAttributes() {
    return numericalAttributes;
  }

  /**
   * Returns the number of categorical attributes.
   *
   * @return the number of categorical attributes.
   */
  public final int categoricalAttributes() {
    return categoricalAttributes;
  }

  /**
   * Returns the number of attributes (this is equal to
   * {@code numericalAttributes() + categoricalAttributes()}).
   *
   * @return the number of attributes
   */
  public final int attributes() {
    return attributes;
  }

  /**
   * Returns true if the attribute at {@code index} is numerical.
   *
   * @param index the index to check if it is numerical
   * @return true of the attribute is numerical
   */
  public final boolean isNumericalAttribute(int index) {
    Check.index(index, attributes());
    return index < numericalAttributes;
  }

  /**
   * Returns true if this schema contains numerical attribute.
   *
   * @return true if this schema contains numerical attribute.
   */
  public final boolean hasNumericalAttributes() {
    return numericalAttributes() > 0;
  }

  /**
   * Returns true if this schema contains categorical attribute.
   *
   * @return true if this schema contains categorical attribute.
   */
  public final boolean hasCategoricalAttributes() {
    return categoricalAttributes() > 0;
  }

  /**
   * Returns a double array of the numerical attributes in the input.
   *
   * The input must have {@code this} as its schema.
   *
   * @param instances the instances to get numerical attributes from
   * @return a double array of the numerical attributes
   */
  public DoubleArray getNumericalAttributes(Input<? extends Instance> instances) {
    Check.argument(instances.getSchema() == this, ILLEGAL_SCHEMA);
    DoubleArray array = DoubleArray.zeros(instances.size(), numericalAttributes());
    for (int i = 0; i < instances.size(); i++) {
      array.setRow(i, instances.get(i).getNumericalAttributes());
    }
    return array;
  }

  /**
   * Returns a array of the categorical attributes in the input.
   *
   * The input must have {@code this} as its schema.
   *
   * @param instances the instances to get categorical attributes from
   * @return a double array of the categorical attributes
   */
  public Array<Object> getCategoricalAttributes(Input<? extends Instance> instances) {
    Check.argument(instances.getSchema() == this, ILLEGAL_SCHEMA);
    Array<Object> array = Array.empty(instances.size(), categoricalAttributes());
    for (int i = 0; i < instances.size(); i++) {
      array.setRow(i, instances.get(i).getCategoricalAttributes());
    }
    return array;
  }

  /**
   * Returns true if the instance has the same number of numerical and categorical attributes as
   * defined by this schema.
   * 
   * @param ex the instance
   * @return true if the instance is valid
   */
  @Override
  public final boolean isValid(Instance ex) {
    return ex.numericalAttributes() == numericalAttributes
        && ex.categoricalAttributes() == categoricalAttributes;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("MultidimensionalSchema\n");
    for (int i = 0; i < attributes(); i++) {
      sb.append("  ").append(i).append(": ").append(getAttributeName(i));
      if (isNumericalAttribute(i)) {
        sb.append(" (numeric)");
      } else {
        sb.append(" (categoric)");
      }
      sb.append("\n");
    }
    return sb.toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;

    MultidimensionalSchema schema = (MultidimensionalSchema) o;
    return numericalAttributes == schema.numericalAttributes
        && categoricalAttributes == schema.categoricalAttributes;
  }

  @Override
  public int hashCode() {
    int result = numericalAttributes;
    result = 31 * result + categoricalAttributes;
    return result;
  }

  private class MultidimensionalBuilder implements InstanceBuilder {

    private Object[] categoricalAttributes;
    private double[] numericalAttributes;

    public MultidimensionalBuilder() {
      numericalAttributes = new double[MultidimensionalSchema.this.numericalAttributes];
      categoricalAttributes = new Object[MultidimensionalSchema.this.categoricalAttributes];
    }

    @Override
    public InstanceBuilder set(String col, double value) {
      int i = containsAttribute(col);
      Check.argument(i >= 0, "Can't find %s", col);
      return set(i, value);
    }

    @Override
    public InstanceBuilder set(int i, double value) {
      Check.index(i, numericalAttributes.length);
      numericalAttributes[i] = value;
      return this;
    }

    @Override
    public InstanceBuilder set(String col, Object value) {
      int i = containsAttribute(col);
      Check.argument(i >= 0, "Can't find %s", col);
      return set(i, value);
    }

    @Override
    public InstanceBuilder set(int i, Object value) {
      Check.index(i, categoricalAttributes.length);
      categoricalAttributes[i] = value;
      return this;
    }

    @Override
    public InstanceBuilder setAll(Array<?> categoricalValues) {
      Check.dimension(categoricalAttributes.length, categoricalValues.size());
      for (int i = 0; i < categoricalValues.size(); i++) {
        categoricalAttributes[i] = categoricalValues.get(i);
      }
      return this;
    }

    @Override
    public InstanceBuilder setAll(DoubleArray numericalValues) {
      Check.dimension(numericalAttributes.length, numericalValues.size());
      for (int i = 0; i < numericalValues.size(); i++) {
        numericalAttributes[i] = numericalValues.get(i);
      }
      return this;
    }

    @Override
    public Instance build() {
      Instance instance;
      if (categoricalAttributes.length > 0 && numericalAttributes.length > 0) {
        instance = new ImmutableInstance(categoricalAttributes, numericalAttributes);
      } else if (categoricalAttributes.length > 0) {
        instance = new ImmutableCategoricalInstance(Array.of(categoricalAttributes));
      } else {
        instance = new ImmutableNumericalInstance(DoubleArray.of(numericalAttributes));
      }
      categoricalAttributes = null;
      numericalAttributes = null;
      return instance;
    }
  }
}
