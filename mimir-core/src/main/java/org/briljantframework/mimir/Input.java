package org.briljantframework.mimir;

import java.util.Collection;

/**
 * Input variables
 */
public interface Input<T> extends Collection<T> {

  InputProperties getProperties();

  T get(int row);
}
