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
package org.briljantframework.mimir.weka;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.vector.Convert;
import org.briljantframework.mimir.*;
import org.briljantframework.mimir.classification.AbstractClassifier;
import org.briljantframework.mimir.classification.ClassifierCharacteristic;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class WekaClassifier<T extends weka.classifiers.Classifier>
    extends AbstractClassifier<org.briljantframework.mimir.Instance> {

  private final T classifier;
  private final FastVector names;

  protected WekaClassifier(T classifier, FastVector names, List<?> classes) {
    super(classes);
    this.classifier = classifier;
    this.names = names;
  }

  private static void addValue(Instance instance, int j, Object value) {
    if (Is.NA(value)) {
      instance.setMissing(j);
    } else if (Is.numeric(value)) {
      instance.setValue(j, Convert.to(Number.class, value).doubleValue());
    } else {
      instance.setValue(j, value.toString());
    }
  }

  public T getClassifier() {
    return classifier;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  @Override
  public DoubleArray estimate(org.briljantframework.mimir.Instance record) {
    Instance instance = new Instance(record.size() + 1);
    Instances instances = new Instances("Crap", names, 1);
    instance.setDataset(instances);
    for (int i = 0; i < record.size(); i++) {
      addValue(instance, i, record.get(i));
    }
    instance.setMissing(record.size());
    instances.setClassIndex(record.size());
    try {
      double[] value = getClassifier().distributionForInstance(instance);
      return DoubleArray.of(value);
    } catch (Exception e) {
      DoubleArray p = DoubleArray.zeros(getClasses().size());
      try {
        double value = getClassifier().classifyInstance(instance);
        for (int i = 0; i < getClasses().size(); i++) {
          if (i == value) {
            p.set(i, 1);
          }
        }
        return p;
      } catch (Exception e1) {
        throw new IllegalStateException("Can't make classification", e1);
      }
    }

  }

  public static class Learner<T extends weka.classifiers.Classifier> implements
      Predictor.Learner<org.briljantframework.mimir.Instance, Object, WekaClassifier<T>> {

    private final T classifier;

    public Learner(T classifier) {
      this.classifier = classifier;
    }

    @Override
    public WekaClassifier<T> fit(Input<? extends org.briljantframework.mimir.Instance> x,
        Output<?> y) {
      PropertyPreconditions.checkProperties(getRequiredInputProperties(), x);

      try {
        List<?> classes = Outputs.unique(y);
        FastVector classVector = new FastVector();
        classes.forEach(classVector::addElement);

        @SuppressWarnings("unchecked")
        T copy = (T) weka.classifiers.Classifier.makeCopy(classifier);
        FastVector names = new FastVector();
        List<String> featureNames = x.getProperty(Dataset.FEATURE_NAMES);
        for (Object column : featureNames) {
          // Guess numeric for now
          Attribute element = new Attribute(column.toString());
          names.addElement(element);
        }
        names.addElement(new Attribute("Class", classVector));
        Instances instances = new Instances("dataFrameCopy", names, x.size());
        for (int i = 0; i < x.size(); i++) {
          org.briljantframework.mimir.Instance record = x.get(i);
          Instance instance = new Instance(record.size() + 1);
          instance.setDataset(instances);

          for (int j = 0; j < record.size(); j++) {
            Object value = record.get(j);
            addValue(instance, j, value);
          }
          instance.setValue(record.size(), y.get(String.class, i));
          instances.add(instance);
        }
        instances.setClassIndex(x.getProperty(Dataset.FEATURE_SIZE));
        copy.buildClassifier(instances);
        return new WekaClassifier<>(copy, names, classes);
      } catch (Exception e) {
        throw new IllegalStateException("Unsupported classifier", e);
      }
    }
  }

}
