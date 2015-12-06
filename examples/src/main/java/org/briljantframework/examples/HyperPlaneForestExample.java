package org.briljantframework.examples;

import java.io.FileInputStream;
import java.io.IOException;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.MixedDataFrame;
import org.briljantframework.data.dataframe.transform.ZNormalizer;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.RdsDatasetReader;
import org.briljantframework.mimir.classification.ClassifierValidator;
import org.briljantframework.mimir.classification.EnsembleEvaluator;
import org.briljantframework.mimir.classification.HyperPlaneTree;
import org.briljantframework.mimir.classification.RandomForest;
import org.briljantframework.mimir.evaluation.Evaluator;
import org.briljantframework.mimir.evaluation.Result;
import org.briljantframework.mimir.evaluation.Validator;

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
