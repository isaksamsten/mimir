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
package org.briljantframework.mimir.data;

/**
 * @author Isak Karlsson
 */
public interface TypeKey<T> {

  static <T> TypeKey<T> of(String name, Class<T> cls) {
    return new StringTypeKey<>(cls, name);
  }

  static <T> TypeKey<T> of(String name, Class<T> cls, T defaultValue) {
    return new StringTypeKey<>(cls, name, defaultValue);
  }

  /**
   * Return the class of the given property
   * 
   * @return the class of the property
   */
  Class<T> getType();

  default T defaultValue() {
    return null;
  }

  /**
   * Return the name of the property
   * 
   * @return the property name
   */
  String getName();

  /**
   * Return the description of the property
   * 
   * @return the property description
   */
  String getDescription();

}
