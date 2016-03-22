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
 * @author Isak Karlsson
 */
public final class Properties {

  private final Map<Property<?>, Object> properties;

  public Properties() {
    this.properties = new HashMap<>();
  }

  public Properties(Properties properties) {
    this.properties = new HashMap<>(properties.properties);
  }

  public boolean contains(Property<?> property) {
    return properties.containsKey(property);
  }

  public boolean containsAll(Collection<? extends Property<?>> properties) {
    for (Property<?> property : properties) {
      if (!contains(property)) {
        return false;
      }
    }
    return true;
  }

  public Set<Property<?>> keySet() {
    return Collections.unmodifiableSet(properties.keySet());
  }

  public <T> T get(Property<T> property) {
    Object value = properties.get(property);
    if (value != null) {
      return property.getType().cast(value);
    } else {
      throw new NoSuchElementException("illegal property");
    }
  }

  public <T> void set(Property<T> property, T value) {
    if (value == null || !property.getType().isInstance(value)) {
      throw new IllegalArgumentException();
    }
    properties.put(property, value);
  }
}
