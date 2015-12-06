package org.briljantframework.examples;

import java.util.Random;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.vector.DoubleVector;
import org.briljantframework.data.vector.IntVector;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.classification.EnsembleClassifierMeasure;
import org.briljantframework.mimir.classification.EnsembleEvaluator;
import org.briljantframework.mimir.classification.RandomShapeletForest;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class RandomClassiciationExample {

  public static void main(String[] args) {
    DataFrame data = Datasets.loadSyntheticControl();
    DataFrame xPos = data.drop(0);

    DataFrame.Builder xBuilder = new DataSeriesCollection.Builder(double.class);
    Vector.Builder yBuilder = new IntVector.Builder();
    for (int i = 0; i < xPos.rows(); i++) {
      yBuilder.add(1);
      xBuilder.addRecord(Vectors.transferableBuilder(xPos.loc().getRecord(i)));
    }
    Random random = new Random();
    for (int i = 0; i < xPos.rows(); i++) {
      yBuilder.add(0);
      Vector.Builder r = new DoubleVector.Builder();
      for (int j = 0; j < xPos.columns(); j++) {
        r.loc().set(j, xPos.loc().get(random.nextInt(xPos.rows()), j));
      }
      xBuilder.addRecord(r);
    }

    DataFrame x = xBuilder.build();
    Vector y = yBuilder.build();

    RandomShapeletForest.Learner learner = new RandomShapeletForest.Configurator(100).configure();
    RandomShapeletForest forest = learner.fit(x, y);

    EnsembleClassifierMeasure measure = new EnsembleClassifierMeasure(forest, x, y, x, y);
    System.out.println(measure.getOobErro());



  }

}
