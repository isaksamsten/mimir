package org.mimirframework.examples;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.array.Range;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.ClassifierMeasure;
import org.mimirframework.classification.LogisticRegression;

/**
 * Created by isak on 11/18/15.
 */
public class ClassifierExample {

  public static void main(String[] args) {
    DataFrame iris = Datasets.loadIris();
    DataFrame x = iris.drop("Class");

    // Multiclass setting (3 classes), however, any c > 2 is supported
    Vector y = iris.get("Class");

    LogisticRegression.Learner lrl = new LogisticRegression.Learner();

    IntArray idx = Arrays.shuffle(Range.of(x.rows()));

    // Use the first 100 examples as training
    IntArray trainIdx = idx.get(Range.of(100));

    // And the last 50 examples as testing
    IntArray testIdx = idx.get(Range.of(100, 150));

    // Get the training data (replacing missing values with the column mean)
    DataFrame xTrain = x.loc().getRecord(trainIdx).apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector yTrain = y.loc().get(trainIdx);
    System.out.println(yTrain.toList());

    // Get the testing data
    DataFrame xTest = x.loc().getRecord(testIdx).apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector yTest = y.loc().get(testIdx);

    // Fit the logistic regression model
    LogisticRegression lr = lrl.fit(xTrain, yTrain);

    // Predict the class label of the test instances
    Vector predicted = lr.predict(xTest);

    // Compute the probabilities for each class
    DoubleArray scores = lr.estimate(xTest);

    // Comute some classifier measures (accurac, auc, etc.)
    ClassifierMeasure cm = new ClassifierMeasure(predicted, yTest, scores, lr.getClasses());

    System.out.println(cm.getAccuracy());
    System.out.println(cm.getAreaUnderRocCurve());
    System.out.println(cm.getBrierScore());

    // We can also inspect the parameters
    System.out.println(lr.getOddsRatio("Sepal Width"));
    System.out.println(lr.getLogLoss());
    System.out.println(lr.getParameters());
  }
}
