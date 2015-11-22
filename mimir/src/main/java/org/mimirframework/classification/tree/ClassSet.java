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

package org.mimirframework.classification.tree;


import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson
 */
public final class ClassSet implements Iterable<Example> {

  private static final Random RANDOM = new Random();

  private final Map<Object, Sample> samples;

  private final List<Object> targets;
  private final Vector domain;

  public ClassSet(Vector column, Vector domain) {
    this(domain);
    for (int i = 0; i < column.size(); i++) {
      add(column.loc().get(Object.class, i), i, 1);
    }
  }

  public ClassSet(Vector domain) {
    samples = new HashMap<>();
    targets = new ArrayList<>();
    this.domain = domain;
  }

  public void add(Object target, int index, double weight) {
    Sample sample = samples.get(target);
    if (sample == null) {
      sample = new Sample(target);
      sample.add(new Example(index, weight));
      samples.put(target, sample);
      targets.add(target);
    } else {
      sample.add(new Example(index, weight));
    }
  }

  public void add(Sample sample) {
    Object target = sample.getTarget();
    targets.add(target);
    samples.put(target, sample);
  }

  public Collection<Sample> samples() {
    return samples.values();
  }

  public int size() {
    int i = 0;
    for (Map.Entry<Object, Sample> kv : samples.entrySet()) {
      i += kv.getValue().size();
    }
    return i;
  }

  public Object getMostProbable() {
    double max = Double.NEGATIVE_INFINITY;
    Object target = null;
    for (Map.Entry<Object, Sample> kv : samples.entrySet()) {
      double weight = kv.getValue().getWeight();
      if (weight > max) {
        target = kv.getKey();
        max = weight;
      }
    }
    return target;
  }

  public Vector getDomain() {
    return domain;
  }

  public Sample get(Object target) {
    return samples.get(target);
  }

  public Sample getRandomSample() {
    return samples.get(targets.get(RANDOM.nextInt(targets.size())));
  }

  public boolean isEmpty() {
    return targets.isEmpty();
  }

  public List<Object> getTargets() {
    return Collections.unmodifiableList(targets);
  }

  public DoubleArray getRelativeFrequencies() {
    double size = getTotalWeight();
    double[] rel = new double[samples.size()];
    int i = 0;
    for (Sample c : samples.values()) {
      rel[i++] = c.getWeight() / size;
    }
    return Arrays.newDoubleVector(rel);
  }

  public double getTotalWeight() {
    double size = 0;
    for (Map.Entry<Object, Sample> kv : samples.entrySet()) {
      size += kv.getValue().getWeight();
    }

    return size;
  }

  @Override
  public String toString() {
    return String.format("Examples(%.2f, %d, %d)", getTotalWeight(), getTargetCount(),
        System.identityHashCode(this));
  }

  public int getTargetCount() {
    return targets.size();
  }

  @Override
  public Iterator<Example> iterator() {
    return new Iterator<Example>() {
      Iterator<Sample> sampleIterator = samples.values().iterator();
      Iterator<Example> exampleIterator = sampleIterator.next().iterator();

      @Override
      public boolean hasNext() {
        return exampleIterator != null && exampleIterator.hasNext();
      }

      @Override
      public Example next() {
        Example example = exampleIterator.next();
        if (!exampleIterator.hasNext()) {
          if (!sampleIterator.hasNext()) {
            exampleIterator = null;
          } else {
            exampleIterator = sampleIterator.next().iterator();
          }
        }
        return example;
      }
    };

  }

  public static final class Sample implements Iterable<Example> {

    private final Object target;
    private final ArrayList<Example> examples;
    private double weight;

    private Sample(Object target) {
      this.target = target;
      this.examples = new ArrayList<>();
      this.weight = 0;
    }

    public static Sample create(Object target) {
      return new Sample(target);
    }

    public Object getTarget() {
      return target;
    }

    public Example get(int index) {
      return examples.get(index);
    }

    public int size() {
      return examples.size();
    }

    public boolean isEmpty() {
      return getWeight() == 0;
    }

    public double getWeight() {
      return weight;
    }

    public void add(Example example) {
      examples.add(example);
      weight += example.getWeight();
    }

    public Example getRandomExample() {
      return examples.get(RANDOM.nextInt(examples.size()));
    }

    @Override
    public Iterator<Example> iterator() {
      return examples.iterator();
    }

    @Override
    public String toString() {
      return String.format("Examples(%.2f)", getWeight());
    }
  }

}
