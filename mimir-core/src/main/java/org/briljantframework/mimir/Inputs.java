package org.briljantframework.mimir;



import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;

/**
 * @author Isak Karlsson
 */
public final class Inputs {

  private Inputs() {}

  /**
   * Create a new input based on instance from a {@link DataFrame}.
   * 
   * @param dataFrame the data frame
   * @return a new instance input
   */
  public static Input<Instance> newInput(DataFrame dataFrame) {
    return new DataFrameInput(dataFrame);
  }

  /**
   * Returns an unmodifiable view of the given input.
   * 
   * @param input the input
   * @return an unmodifiable view of the given input
   */
  public static <T> Input<T> unmodifiableInput(Input<? extends T> input) {
    return new AbstractInput<T>() {
      @Override
      public int size() {
        return input.size();
      }

      @Override
      public Properties getProperties() {
        return input.getProperties();
      }

      @Override
      public T get(int index) {
        return input.get(index);
      }
    };
  }


  public static int features(Input<? extends Instance> tabularInput) {
    int size = tabularInput.get(0).size();
    for (int i = 1; i < tabularInput.size(); i++) {
      if (size != tabularInput.get(i).size()) {
        throw new IllegalArgumentException("Not a tabular input");
      }
    }
    return size;
  }

  public static DoubleArray toDoubleArray(Input<? extends Instance> x) {
    int n = x.size();
    int m = x.getProperty(Dataset.FEATURES);
    DoubleArray out = DoubleArray.zeros(n, m);
    for (int i = 0; i < n; i++) {
      Instance v = x.get(i);
      for (int j = 0; j < m; j++) {
        out.set(i, j, v.getAsDouble(j));
      }
    }
    return out;
  }
}
