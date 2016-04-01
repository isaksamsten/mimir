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
package org.briljantframework.mimir.data;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * @author Isak Karlsson
 */
public final class Dataset {
  /**
   * Castable class instance (it is used for 'reified' generic parameters)
   */
  private final static Class<? extends List> CLASS = List.class;

  /**
   * Returns true if the input can be considered a dataset, i.e. has the properties
   * {@link #FEATURE_SIZE} and {@link #FEATURE_TYPES}.
   * 
   * @param input the input
   * @return true if the given input is a dataset.
   */
  public static boolean isDataset(Input<?> input) {
    return input.getProperties().containsAll(Arrays.asList(FEATURE_TYPES, FEATURE_SIZE))
        && input.getProperty(FEATURE_TYPES).size() == input.getProperty(FEATURE_SIZE);
  }

  /**
   * Returns true if all features (as defined by the {@link #FEATURE_TYPES} property) are numeric.
   *
   * @param input the input
   * @return true if all features are numeric
   */
  public static boolean isAllNumeric(Input<?> input) {
    return isAllNumeric(input.getProperty(Dataset.FEATURE_TYPES));
  }

  private static boolean isAllNumeric(Collection<? extends Class<?>> collection) {
    return collection.stream().allMatch(Number.class::isAssignableFrom);
  }

  /**
   * This property denotes the number of features in an input dataset. Often, this is the number of
   * columns.
   */
  public static final TypeKey<Integer> FEATURE_SIZE = new TypeKey<Integer>() {
    @Override
    public Class<Integer> getType() {
      return Integer.class;
    }

    @Override
    public String getName() {
      return "Number of input features.";
    }

    @Override
    public String getDescription() {
      return "Number of input features";
    }

    @Override
    public boolean validate(Integer value) {
      return value != null && value > 0;
    }
  };

  /**
   * This property denotes the feature types of a tabular dataset
   */
  public static final TypeKey<List<Class<?>>> FEATURE_TYPES = new TypeKey<List<Class<?>>>() {

    @Override
    @SuppressWarnings("unchecked")
    public Class<List<Class<?>>> getType() {
      return (Class<List<Class<?>>>) CLASS;
    }

    @Override
    public String getName() {
      return "List of feature types";
    }

    @Override
    public String getDescription() {
      return "List of feature types";
    }
  };

  public static final TypeKey<List<String>> FEATURE_NAMES = new TypeKey<List<String>>() {
    @Override
    @SuppressWarnings("unchecked")
    public Class<List<String>> getType() {
      return (Class<List<String>>) CLASS;
    }

    @Override
    public String getName() {
      return "List of feature names";
    }

    @Override
    public String getDescription() {
      return "List of feature names";
    }
  };
}
