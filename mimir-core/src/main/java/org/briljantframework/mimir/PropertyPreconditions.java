package org.briljantframework.mimir;

import java.util.Collection;

/**
 * @author Isak Karlsson
 */
public final class PropertyPreconditions {
  private PropertyPreconditions() {}

  public static void checkProperties(Collection<? extends Property<?>> required,
      Properties properties) {
    for (Property<?> property : required) {
      if (!properties.contains(property)) {
        throw new IllegalArgumentException(
            String.format("Required property '%s' not set.", property.getName()));
      }
    }
  }

  public static void checkProperties(Collection<? extends Property<?>> required, Input<?> input) {
    checkProperties(required, input.getProperties());
  }
}
