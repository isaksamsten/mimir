package org.mimirframework.examples;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.IntArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.EnsembleEvaluator;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;
import org.mimirframework.evaluation.partition.FoldPartitioner;

/**
 * @author Isak Karlsson
 */
public class RandomForestExample {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class");
    Vector y = iris.get("Class");
    System.out.println(y);

    System.out.println(x);
    IntArray f = Arrays.newIntVector(10, 2, 3);
    Validator<RandomForest> classifierValidator =
        new ClassifierValidator<>(new FoldPartitioner(10));
    classifierValidator.add(EnsembleEvaluator.INSTANCE);
    for (int i = 0; i < f.size(); i++) {
      RandomForest.Learner forest =
          new RandomForest.Configurator(100).setMaximumFeatures(f.get(i)).configure();
      Result result = classifierValidator.test(forest, x, y);
      System.out.println(result.getMeasures());
    }
  }
}
