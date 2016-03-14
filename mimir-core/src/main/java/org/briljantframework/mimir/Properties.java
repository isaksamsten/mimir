package org.briljantframework.mimir;

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
