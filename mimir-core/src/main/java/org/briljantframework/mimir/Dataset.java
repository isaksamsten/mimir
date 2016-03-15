package org.briljantframework.mimir;

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
   * Returns true if all classes in the collection is assign from {@link Number}.
   * 
   * @param collection the collection of classes
   * @return true if all classes is assignable from {@link Number}
   */
  public static boolean isAllNumeric(Collection<? extends Class<?>> collection) {
    return collection.stream().allMatch(Number.class::isAssignableFrom);
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

  /**
   * This property denotes the number of features in an input dataset. Often, this is the number of
   * columns.
   */
  public static final Property<Integer> FEATURE_SIZE = new Property<Integer>() {
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
  };

  /**
   * This property denotes the feature types of a tabular dataset
   */
  public static final Property<List<Class<?>>> FEATURE_TYPES = new Property<List<Class<?>>>() {


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

  public static final Property<List<String>> FEATURE_NAMES = new Property<List<String>>() {
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
