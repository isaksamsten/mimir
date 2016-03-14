package org.briljantframework.mimir;

/**
 * @author Isak Karlsson
 */
public interface Property<T> {

  Class<T> getType();

  String getName();

  String getDescription();
}
