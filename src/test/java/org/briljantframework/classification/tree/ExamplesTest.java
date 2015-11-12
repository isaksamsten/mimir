/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.briljantframework.classification.tree;

public class ExamplesTest {

  // @Test
  // public void testSplit() throws Exception {
  // long start = System.currentTimeMillis();
  // Container<DenseDataset, DefaultTarget> container = new CSVInputStream(new FileInputStream
  // ("erlang/adeb-rr/deps/rr/data/connect-4.txt")).read(DenseDataset.FACTORY,
  // DefaultTarget.FACTORY);
  // System.out.println(System.currentTimeMillis() - start);
  //
  // Target target = container.getTarget();
  //
  // System.out.println(container);
  // Random random = new Random();
  // start = System.currentTimeMillis();
  //
  // Examples examples = Examples.create();
  // for (int i = 0; i < target.size(); i++) {
  // if (random.nextDouble() > 0.3)
  // examples.add(target.getValue(i), i, 1.0f);
  // }
  //
  // System.out.println(examples.getTotalWeight());
  // System.out.println(System.currentTimeMillis() - start);
  //
  // start = System.currentTimeMillis();
  //
  // DecisionTree tree = DecisionTree.withSplitter(RandomSplitter
  // .withMaximumFeatures(2)
  // .withCriterion(Gain.GINI))
  // .create();
  // System.out.println(Evaluators.crossValidation(tree, container, 10));
  // // DecisionTree.Classifier mdl = tree.fit(storage);
  //
  // // mdl = tree.fit(storage);
  // // mdl = tree.fit(storage);
  //
  // // List<Prediction> predictions = mdl.predict(storage.getDataset());
  // // int correct = 0;
  // // for (int i = 0; i < predictions.size(); i++) {
  // //// System.out.println(predictions.get(i) + " :: " + storage.getTarget().getValue(i));
  // // if (predictions.get(i).getValue().equals(storage.getTarget().getValue(i))) {
  // // correct++;
  // // }
  // // }
  // // System.out.println((double) correct / storage.rows());
  // // System.out.println(System.currentTimeMillis() - start);
  // }
  //
  // @Test
  // public void testIterator() throws Exception {
  // Container<DenseDataset, DefaultTarget> container = new CSVInputStream(new FileInputStream
  // ("erlang/adeb-rr/deps/rr/data/connect-4.txt")).read(DenseDataset.FACTORY,
  // DefaultTarget.FACTORY);
  // Examples examples = Examples.fromContainer(container);
  // int i = 0;
  // for (Example e : examples) {
  // container.getTarget().getValue(e.getIndex());
  // i++;
  // }
  //
  // Assert.assertEquals(container.rows(), i);
  // }
}
