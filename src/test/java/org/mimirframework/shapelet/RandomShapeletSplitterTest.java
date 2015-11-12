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

package org.mimirframework.shapelet;

import org.junit.Test;

public class RandomShapeletSplitterTest {

  @Test
  public void testSplitter() throws Exception {
    // DataSeriesInputStream tsis = new DataSeriesInputStream(new
    // FileInputStream("erlang/synthetic_control_TRAIN
    // .txt"));
    // Container<Frame, DefaultTarget> container = tsis.read(Frame.FACTORY, DefaultTarget.FACTORY);
    //
    // RandomShapeletSplitter splitter =
    // RandomShapeletSplitter.withDistance(EarlyAbandonSlidingDistance.create
    // (Distance.EUCLIDEAN)).create();
    //
    // Examples examples = Examples.fromContainer(container);
    // Tree.Split<ShapeletThreshold> split = splitter.find(examples, container);
    // System.out.println(split.getThreshold().getDistance());
    // System.out.println(split);
    //
    //
    // ShapeletTree tree = new ShapeletTree(splitter);
    //
    // ShapeletTree.Classifier model = tree.fit(container);
    //
    // Ensemble.Member<Frame> member = ShapeletTree.withSplitter(RandomShapeletSplitter.withDistance
    // (EarlyAbandonSlidingDistance.create(Distance.EUCLIDEAN)));
    // Ensemble<Frame> ensemble = Ensemble.withMember(member).create();
    //
    //
    // System.out.println(model.predict(container.getDataset().getEntry(6)));
    // System.out.println(container.getTarget(6));
    //
    // KNearestNeighbors oneNearestNeighbours = KNearestNeighbors
    // .withNeighbors(6)
    // .withDistance(DynamicTimeWarping
    // .withDistance(Distance.EUCLIDEAN)
    // .withConstraint(6))
    // .create();
    //
    //
    // Evaluators.splitValidation(ensemble, container, 0.3);
    // Evaluators.splitValidation(oneNearestNeighbours, container, 0.3);

  }

  @Test
  public void testFindThresholdBasedOnDistance() throws Exception {
    // List<ExampleDistance> ed = new ArrayList<>();
    // ed.add(ExampleDistance.create(0, new Example(0, 1)));
    // ed.add(ExampleDistance.create(0, new Example(1, 1)));
    // ed.add(ExampleDistance.create(0, new Example(2, 1)));
    // ed.add(ExampleDistance.create(0, new Example(3, 1)));
    // ed.add(ExampleDistance.create(0, new Example(4, 1)));
    // ed.add(ExampleDistance.create(0, new Example(5, 1)));
    // ed.add(ExampleDistance.create(1.1389165862978277, new Example(6, 1)));
    //
    //
    // Target.Factory<DefaultTarget> f = DefaultTarget.FACTORY;
    // Target.Builder<DefaultTarget> b = f.createBuilder(new CategoricHeader("hello",
    // Type.CATEGORIC_TARGET));
    // b.createTarget("a");
    // b.createTarget("b");
    // b.createTarget("b");
    // b.createTarget("b");
    // b.createTarget("b");
    // b.createTarget("b");
    // b.createTarget("b");
    // Target target = b.create();
    //
    // Frame frame = Frame.FACTORY.createBuilder(Arrays.asList(new NumericHeader("he",
    // Type.DOUBLE))).create();
    //
    // Examples examples = Examples.create();
    // Examples.Sample a = Examples.Sample.create(target.getValue(0));
    // a.add(new Example(0, 1));
    // examples.add(a);
    //
    // Examples.Sample bb = Examples.Sample.create(target.getValue(1));
    // bb.add(new Example(1, 1));
    // bb.add(new Example(2, 1));
    // bb.add(new Example(3, 1));
    // bb.add(new Example(4, 1));
    //
    // bb.add(new Example(5, 1));
    // bb.add(new Example(6, 1));
    // examples.add(bb);
    //
    // RandomShapeletSplitter splitter =
    // RandomShapeletSplitter.withDistance(EarlyAbandonSlidingDistance.create
    // (Distance.EUCLIDEAN)).create();
    // splitter.findBestThreshold(ed, examples, Container.create(frame, target), 3);
  }

  @Test
  public void testFindTresholdUsingParser() throws Exception {
    // String str = "0.0(1,3.0) 0.0043247258021958486(1,1.0) 0.014283299948135524(1,1.0)
    // 0.014283299948135627(1,
    // 2.0) 0.019637546522799212(1,2.0) 0.01963770410523272(1,1.0) 0.023176866042376454(1,2.0)
    // 0.023176993270098426(1,
    // 1.0) 0.03365034967334447(1,1.0) 0.037383690736357465(1,3.0) 0.04616066853208171(1,3.0)
    // 0.05088832979639352(1,
    // 1.0) 0.06135713259230081(1,1.0) 0.06135748801923446(1,3.0) 0.08750985687234046(1,2.0)
    // 0.09381356466888063(1,
    // 2.0) 0.09381356466888063(2,1.0) 0.09381356466888063(2,1.0) 0.09381356466888065(1,1.0)
    // 0.1041169281501413(1,
    // 2.0) 0.13544308147499695(1,1.0) 0.15500299753850172(1,1.0)";
    // String str = "0.0(1,1.0) 0.0021139244145406346(2,2.0) 0.002113933608833587(2,4.0)
    // 0.006075244225533394(2,
    // 1.0) 0.009716919124027851(2,1.0) 0.012169465713969752(1,1.0) 0.012169494638987827(1,1.0)
    // 0.012169513617948878(1,
    // 1.0) 0.012169516494447504(1,1.0) 0.013076007029446278(1,2.0) 0.013076010212583216(1,1.0)
    // 0.013076010651516643(2,
    // 1.0) 0.013076016256105645(1,1.0) 0.0170298041445876(1,1.0) 0.0176525031080276(1,1.0)
    // 0.02878612712775304(2,
    // 2.0) 0.029295977142341566(1,2.0) 0.030844353506909054(2,1.0) 0.031536769085919736(1,1.0)
    // 0.03153682310857459(2,
    // 1.0) 0.03460479727845928(2,3.0) 0.045529843904497394(2,2.0) 0.053154042723057564(1,2.0)
    // 0.05924477848040153(2,
    // 3.0) 0.059245154231763714(2,2.0) 0.05972769065777582(1,2.0) 0.07484989050806876(1,1.0)
    // 0.07546067087869678(2,
    // 1.0) 0.0945381725452216(2,3.0) 0.09592385536832064(2,1.0) 0.09592385536832065(1,1.0)
    // 0.09592385536832065(2,
    // 1.0) 0.09592385536832065(2,1.0) 0.09592385536832067(1,2.0) 0.09592385536832067(1,2.0)
    // 0.1020070032681784(2,
    // 1.0) 0.11347885047107395(1,2.0) 0.13861453122613607(2,1.0) 0.13862771848423672(1,1.0)
    // 0.13995827912349387(1,
    // 1.0) 0.1419812678073083(1,2.0) 0.15007960469833384(1,1.0) 0.16748649985645206(1,2.0)
    // 0.18186321889397275(1,
    // 1.0) 0.19374175049157513(2,1.0)";
    // String str = "0.0(1,1.0) 0.27802778067430056(2,2.0) 0.3316206262210656(2,1.0)
    // 0.3573963879322891(2,
    // 1.0) 0.36507299379463887(2,1.0) 0.36698304743411525(2,2.0) 0.3701566329723696(2,1.0)
    // 0.37234372650564973(2,
    // 1.0) 0.3798982725415759(2,1.0) 0.3886211819765626(2,1.0) 0.3905954615215281(2,1.0)
    // 0.39109482856279437(2,
    // 2.0) 0.3960217426480879(2,1.0) 0.4015223651445833(2,1.0) 0.4060438443208667(2,1.0)
    // 0.40992126364578807(2,
    // 2.0) 0.4149996655264401(2,2.0) 0.41998297220572917(2,1.0) 0.42557503829593274(2,1.0)
    // 0.44273221717131067(2,
    // 2.0) 0.44601196917788016(2,1.0) 0.45914033461194914(2,2.0) 0.5044614865891399(2,2.0)
    // 0.5427903866420823(2,
    // 1.0) 0.7512105269258049(2,1.0) 0.9619947433602803(2,1.0)";
    // Pattern p = Pattern.compile("(.*)\\((.*),(.*)\\)");
    //
    //
    // Target.Factory<DefaultTarget> f = DefaultTarget.FACTORY;
    // Header types = new CategoricHeader("hello", Type.CATEGORIC_TARGET);
    // Target.Builder<DefaultTarget> b = f.createBuilder(types);
    // Examples examples = Examples.create();
    // List<ExampleDistance> ed = new ArrayList<>();
    //
    // int id = 0;
    // for (String part : str.split(" ")) {
    // Matcher m = p.matcher(part);
    // m.matches();
    // double distance = Double.parseDouble(m.group(1));
    // double weight = Double.parseDouble(m.group(3));
    // String target = m.group(2);
    //
    // Value value = types.createValue(target);
    // b.add(value);
    // examples.add(value, id, weight);
    // ed.add(ExampleDistance.create(distance, new Example(id, weight)));
    //
    // id += 1;
    // }
    //
    // Frame frame = Frame.FACTORY.createBuilder(Arrays.asList(new NumericHeader("he",
    // Type.DOUBLE))).create();
    // Target target = b.create();
    // RandomShapeletSplitter splitter =
    // RandomShapeletSplitter.withDistance(EarlyAbandonSlidingDistance.create
    // (Distance.EUCLIDEAN)).create();
    // splitter.findBestThreshold(ed, examples, Container.create(frame, target), 3);

  }
}
