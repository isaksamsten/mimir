package org.briljantframework.mimir.supervised;

import org.briljantframework.mimir.data.TypeKey;
import org.briljantframework.mimir.data.TypeMap;

/**
 * Created by isak on 3/29/16.
 */
public abstract class AbstractLearner<In, Out, P extends Predictor<In, Out>>
    implements Predictor.Learner<In, Out, P> {

  private final TypeMap parameters;

  protected AbstractLearner(TypeMap parameters) {
    this.parameters = new TypeMap(parameters);
  }

  protected AbstractLearner() {
    this.parameters = new TypeMap();
  }

  @Override
  public final <T> void set(TypeKey<T> key, T value) {
    parameters.set(key, value);
  }

  @Override
  public final <T> T get(TypeKey<T> key) {
    return parameters.get(key);
  }

  @Override
  public final TypeMap getParameters() {
    return parameters;
  }

  @Override
  public String toString() {
    return getClass().getSimpleName() + "{ parameters: " + getParameters() + "}";
  }
}
