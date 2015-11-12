package org.briljantframework.supervised;

import java.util.Set;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface Predictor {

  /**
   * Return a vector of predictions for the rows in the given data frame.
   * 
   * @param x the data frame
   * @return a vector of predictions
   */
  Vector predict(DataFrame x);

  /**
   * Get a set of characteristics for this particular predictor
   *
   * @return the set of characteristics
   */
  Set<Characteristic> getCharacteristics();

  interface Learner<P extends Predictor> {
    P fit(DataFrame x, Vector y);
  }

  interface Configurator<C extends Learner<? extends Predictor>> {
    C configure();
  }
}
