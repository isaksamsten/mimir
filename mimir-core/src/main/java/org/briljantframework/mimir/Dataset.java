package org.briljantframework.mimir;

import java.util.List;

/**
 * Created by isak on 3/11/16.
 */
public final class Dataset {
  public static final InputProperty<Integer> FEATURES = () -> Integer.class;
  public static final InputProperty<List> TYPES = () -> List.class;
}
