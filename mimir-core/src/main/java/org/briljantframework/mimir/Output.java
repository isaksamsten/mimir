package org.briljantframework.mimir;

import java.util.Collection;

/**
 * Output variables
 */
public interface Output<T> extends Collection<T> {

  T get(int i);
}
