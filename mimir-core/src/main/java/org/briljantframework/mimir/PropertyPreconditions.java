package org.briljantframework.mimir;

import java.util.Collection;

/**
 * @author Isak Karlsson
 */
public final class PropertyPreconditions {
  private PropertyPreconditions() {}

  public static void checkParameters(Collection<? extends Property<?>> required,
      Properties properties) {
    for (Property<?> property : required) {
      if (!properties.contains(property)) {
        throw new IllegalInputException(required, properties.keySet());
      }
    }
  }

  public static void checkParameters(Collection<? extends Property<?>> required, Input<?> input) {
    checkParameters(required, input.getProperties());
  }
}
