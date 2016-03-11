package org.briljantframework.mimir;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by isak on 3/11/16.
 */
public final class InputProperties {

  private final Map<InputProperty<?>, Object> properties = new HashMap<>();

  public <T> T get(InputProperty<T> property) {
    Object value = properties.get(property);
    if (value != null) {
      return property.getType().cast(value);
    } else {
      return null;
    }
  }

  public <T> void set(InputProperty<T> property, T value) {
    if (!property.getType().isInstance(value)) {
      throw new IllegalArgumentException();
    }
    properties.put(property, value);
  }
}
