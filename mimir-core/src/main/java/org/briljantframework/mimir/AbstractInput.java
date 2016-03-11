package org.briljantframework.mimir;

import java.util.AbstractCollection;

/**
 * Created by isak on 3/11/16.
 */
public abstract class AbstractInput<T> extends AbstractCollection<T> implements Input<T> {

  private final InputProperties properties;

  protected AbstractInput(InputProperties properties) {
    this.properties = properties;
  }

  protected AbstractInput() {
    this.properties = new InputProperties();
  }

  @Override
  public final InputProperties getProperties() {
    return properties;
  }
}
