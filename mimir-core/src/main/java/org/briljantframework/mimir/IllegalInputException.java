package org.briljantframework.mimir;

import java.util.Collection;
import java.util.Collections;

/**
 * @author Isak
 */
public class IllegalInputException extends IllegalArgumentException {

  private final Collection<? extends Property<?>> required;
  private final Collection<? extends Property<?>> actual;

  public IllegalInputException(Collection<? extends Property<?>> required,
      Collection<? extends Property<?>> actual) {
    this.required = required;
    this.actual = actual;
  }

  public Collection<Property<?>> getRequired() {
    return Collections.unmodifiableCollection(required);
  }

  public Collection<Property<?>> getActual() {
    return Collections.unmodifiableCollection(actual);
  }
}
