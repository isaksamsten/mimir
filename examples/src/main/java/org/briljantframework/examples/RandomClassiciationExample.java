/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
/**
 * This copy of Woodstox XML processor is licensed under the
 * Apache (Software) License, version 2.0 ("the License").
 * See the License for details about distribution rights, and the
 * specific rights regarding derivate works.
 *
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/
 *
 * A copy is also included in the downloadable source code package
 * containing Woodstox, in file "ASL2.0", under the same directory
 * as this file.
 */
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
