package org.briljantframework.mimir.classification.tune;

import java.util.function.BiConsumer;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
class UpdatableEnumerationParameter<T, V> implements UpdatableParameter<T> {

  private final V[] enumeration;
  private final BiConsumer<T, V> updater;

  public UpdatableEnumerationParameter(BiConsumer<T, V> updater, V[] enumeration) {
    this.enumeration = enumeration;
    this.updater = updater;
  }

  @Override
  public ParameterUpdator<T> updator() {
    return new ParameterUpdator<T>() {
      private int current = 0;

      @Override
      public boolean hasUpdate() {
        return current < enumeration.length;
      }

      @Override
      public Object update(T toUpdate) {
        V value = enumeration[current++];
        updater.accept(toUpdate, value);
        return value;
      }
    };
  }
}
