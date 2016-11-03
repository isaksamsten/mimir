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
 * Properties is a heterogeneous key-value store where the keys encode the class of the value.
 * 
 * @author Isak Karlsson
 */
public final class Properties {

  private final Map<Property<?>, Object> map;

  public Properties() {
    this.map = new IdentityHashMap<>();
  }

  public Properties(Properties properties) {
    this.map = new IdentityHashMap<>(properties.map);
  }

  /**
   * Return {@code true} if the container contains the specified property.
   * 
   * @param property the property
   * @return true if property exists
   */
  public boolean contains(Property<?> property) {
    return map.containsKey(property);
  }

  /**
   * Returns {@code true} if the container contains all properties in the given collection.
   * 
   * @param properties the properties to check for
   * @return true if the properties exist
   */
  public boolean containsAll(Collection<? extends Property<?>> properties) {
    for (Property<?> property : properties) {
      if (!contains(property)) {
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
  public Set<Property<?>> keySet() {
    return Collections.unmodifiableSet(map.keySet());
  }

  /**
   * Get the value of the specified property.
   * 
   * @param property the key
   * @param <T> the type of the return value
   * @return the value of the specified key
   * @throws NoSuchElementException if the property does not exist
   */
  public <T> T get(Property<T> property) {
    Object value = map.get(property);
    if (value != null) {
      return property.getType().cast(value);
    } else {
      throw new NoSuchElementException("key not found");
    }
  }

  /**
   * Get the value of the specified key or the default value.
   *
   * @param property the key
   * @param defaultValue the default value
   * @param <T> the type of the return value
   * @return the value of the specified key or the default value
   * @throws NullPointerException if the default value is null
   */
  public <T> T getOrDefault(Property<T> property, T defaultValue) {
    if (defaultValue == null) {
      throw new NullPointerException();
    }
    Object value = map.getOrDefault(property, defaultValue);
    return property.getType().cast(value);
  }

  /**
   * Get the value of the specified key or the default value for the key if specified.
   * 
   * @param property the key
   * @param <T> the type of return value
   * @return the value or the default value for the key
   * @throws NullPointerException if the default value is null
   */
  public <T> T getOrDefault(Property<T> property) {
    return getOrDefault(property, property.defaultValue());
  }

  /**
   * Associate the specified key and value, validating the class and properties of the value based
   * on the specified key.
   * 
   * @param property the key
   * @param value the value
   * @param <T> the type of the value
   * @throws NullPointerException if the value is null
   * @throws IllegalArgumentException if value is not an instance suitable for the key or the key
   *         fail to validate the key
   */
  public <T> void set(Property<T> property, T value) {
    if (value == null) {
      throw new NullPointerException("null values are not allowed");
    } else if (!property.getType().isInstance(value)) {
      throw new IllegalArgumentException(
          "The given value is not an instance of the class specified by the key.");
    } else if (!property.validate(value)) {
      throw new IllegalArgumentException(
          "The given value does not conform with the validation requirements of the key");
    } else {
      map.put(property, value);
    }
  }

  @Override
  public String toString() {
    return map.toString();
  }
}
