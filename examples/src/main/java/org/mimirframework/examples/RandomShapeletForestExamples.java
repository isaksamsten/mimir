package org.mimirframework.examples;

import java.io.FileInputStream;
import java.io.IOException;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.DatasetReader;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.MatlabDatasetReader;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.EnsembleEvaluator;
import org.mimirframework.classification.RandomShapeletForest;
import org.mimirframework.classification.ShapeletTree;
import org.mimirframework.evaluation.Evaluator;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;

/**
 * Created by isak on 11/13/15.
 */
public class RandomShapeletForestExamples {

  public static void main(String[] args) throws IOException {
    // This is a dataset which we load
    DataFrame data = DataFrames.permuteRecords(Datasets.loadSyntheticControl());

    // See: loadDatasetExample() for an example of how to load a dataset
    // DataFrame data = DataFrames.permuteRecords(loadDatasetExample());

    // The first column of the dataset contains the class, so we drop it and then extract it
    DataFrame x = data.drop(0);
    Vector y = data.get(0);

    Validator<RandomShapeletForest> cv = ClassifierValidator.crossValidation(10);
    cv.add(Evaluator.foldOutput(i -> System.out.printf("Fold: %d\n", i)));
    cv.add(EnsembleEvaluator.INSTANCE);

    // Initialize a random shapelet forest configurator; 100 trees
    RandomShapeletForest.Configurator config = new RandomShapeletForest.Configurator(100);

    // Use information gain
    config.setAssessment(ShapeletTree.Learner.Assessment.IG);

    // The minimum shapelet length is 2.5% of the time series length
    config.setLowerLength(0.025);
    config.setUpperLength(1.0);

    // Sample 10 shapelets at each node
    config.setMaximumShapelets(10);
    RandomShapeletForest.Learner forest = config.configure();

    // Evaluate the classifier
    Result result = cv.test(forest, x, y);

    // Note that precision and recall is not implemented yet
    System.out.println("Results averaged over 10-fold cross-validation");
    DataFrame measures = result.getMeasures();
    System.out.println(measures.mean());
  }

  public static DataFrame loadDatasetExample() throws IOException {
    // Dataset can be found here: http://www.cs.ucr.edu/~eamonn/time_series_data/
    String trainFile = "/home/isak/Projects/datasets/dataset/Gun_Point/Gun_Point_TRAIN";
    String testFile = "/home/isak/Projects/datasets/dataset/Gun_Point/Gun_Point_TEST";
    try (DatasetReader train = new MatlabDatasetReader(new FileInputStream(trainFile));
        DatasetReader test = new MatlabDatasetReader(new FileInputStream(testFile))) {
      DataFrame.Builder dataset = new DataSeriesCollection.Builder(double.class);
      dataset.readAll(train);
      dataset.readAll(test);
      return dataset.build();
    }
  }

}
