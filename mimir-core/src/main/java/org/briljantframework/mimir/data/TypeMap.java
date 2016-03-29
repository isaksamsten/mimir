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

import java.util.*;

/**
 * Properties is a heterogenoues key-value store where the keys encode the class of the value.
 * 
 * @author Isak Karlsson
 */
public final class TypeMap {

  private final Map<TypeKey<?>, Object> map;

  public TypeMap() {
    this.map = new HashMap<>();
  }

  public TypeMap(TypeMap typeMap) {
    this.map = new HashMap<>(typeMap.map);
  }

  /**
   * Return {@code true} if the container contains the specified property.
   * 
   * @param typeKey the property
   * @return true if property exists
   */
  public boolean contains(TypeKey<?> typeKey) {
    return map.containsKey(typeKey);
  }

  /**
   * Returns {@code true} if the container contains all properties in the given collection.
   * 
   * @param properties the properties to check for
   * @return true if the properties exist
   */
  public boolean containsAll(Collection<? extends TypeKey<?>> properties) {
    for (TypeKey<?> typeKey : properties) {
      if (!contains(typeKey)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Return a set of all properties in this container.
   * 
   * @return a set of properties
   */
  public Set<TypeKey<?>> keySet() {
    return Collections.unmodifiableSet(map.keySet());
  }

  /**
   * Get the value of the specified property.
   * 
   * @param typeKey the key
   * @param <T> the type of the return value
   * @return the value of the specified key
   * @throws NoSuchElementException if the property does not exist
   */
  public <T> T get(TypeKey<T> typeKey) {
    Object value = map.get(typeKey);

    // try the default value if the value is missing
    if (value == null) {
      value = typeKey.defaultValue();
    }

    if (value != null) {
      return typeKey.getType().cast(value);
    } else {
      throw new NoSuchElementException("illegal property");
    }
  }

  public <T> void set(TypeKey<T> typeKey, T value) {
    if (value == null) {
      throw new NullPointerException("null values are note allowed");
    } else if (!typeKey.getType().isInstance(value)) {
      throw new IllegalArgumentException(
          "The given value is not an instance of the class specified by the key.");
    }
    map.put(typeKey, value);
  }

  @Override
  public String toString() {
    return map.toString();
  }
}
