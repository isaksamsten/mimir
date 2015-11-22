package org.mimirframework.weka;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.junit.Test;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.evaluation.Validator;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

import static org.junit.Assert.*;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class WekaClassifierTest {

  @Test
  public void testWekaClassifier() throws Exception {
    J48 tree = new J48();
    WekaClassifier.Learner<?> f = new WekaClassifier.Learner<>(tree);

    Validator<Classifier> cv = ClassifierValidator.crossValidator(10);
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class");
    Vector y = iris.get("Class");

    System.out.println(cv.test(f, x, y).getMeasures().mean());



  }
}
