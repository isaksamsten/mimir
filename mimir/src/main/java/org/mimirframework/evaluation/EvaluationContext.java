package org.mimirframework.evaluation;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.evaluation.partition.Partition;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface EvaluationContext<P extends Predictor> {

  Partition getPartition();

  Vector getPredictions();

  DoubleArray getEstimates();

  P getPredictor();

//  // TODO: measure should depend on P
//  <T extends Measure, C extends Measure.Builder<T>> C getOrDefault(Class<T> measure,
//      Supplier<C> supplier);
//
//  List<Measure> getMeasures();

  MeasureCollection getMeasureCollection();

}
