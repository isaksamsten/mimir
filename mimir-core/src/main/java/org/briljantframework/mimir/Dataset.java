package org.briljantframework.mimir;

import java.util.Collection;
import java.util.List;

/**
 * @author Isak Karlsson
 */
public final class Dataset {
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
   * This property denotes the number of features in an input dataset. Often, this is the number of
   * columns.
   */
  public static final Property<Integer> FEATURES = new Property<Integer>() {
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
