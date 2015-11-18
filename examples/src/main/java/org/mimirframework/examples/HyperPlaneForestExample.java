package org.mimirframework.examples;

import java.io.FileInputStream;
import java.io.IOException;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.MixedDataFrame;
import org.briljantframework.data.dataframe.transform.ZNormalizer;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.RdsDatasetReader;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.EnsembleEvaluator;
import org.mimirframework.classification.HyperPlaneTree;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.evaluation.Evaluator;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;

/**
 * @author Isak Karlsson
 */
public class HyperPlaneForestExample {
  public static void main(String[] args) throws IOException {

    // DataFrame data = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame data = Datasets.load(MixedDataFrame.Builder::new,
        new RdsDatasetReader(new FileInputStream("/home/isak/Desktop/image-segmentation.txt")));
    String classCol = "REGION-CENTROID-COL";
    DataFrame x = new ZNormalizer().fitTransform(data.drop(classCol).apply(v -> v.set(v.where(Is::NA), v.mean())));
    Vector y = data.get(classCol);
    System.out.println(x);
    System.out.println(y);
    Validator<RandomForest> cv = ClassifierValidator.crossValidator(10);
    cv.add(EnsembleEvaluator.INSTANCE);
    cv.add(Evaluator.foldOutput(i -> String.format("%d", i)));

    RandomForest.Configurator configurator = new RandomForest.Configurator(1000);
    configurator
        .setBaseLearner(((classSet, classes) -> new HyperPlaneTree.Learner(classSet, classes, 10)));
    RandomForest.Learner hpRf = configurator.configure();

    RandomForest.Learner rf = configurator.setBaseLearner(null).configure();
    Result hpR = cv.test(hpRf, x, y);
    System.out.println(hpR.getMeasures().mean());
    // Result rfR = cv.test(rf, x, y);
    // System.out.println(hpR.getMeasures().mean().minus(rfR.getMeasures().mean()));

  }
}
