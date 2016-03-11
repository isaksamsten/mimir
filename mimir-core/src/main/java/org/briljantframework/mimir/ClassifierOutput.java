package org.briljantframework.mimir;

import java.util.AbstractCollection;
import java.util.Arrays;
import java.util.Iterator;

/**
 * Created by isak on 3/9/16.
 */
public class ClassifierOutput extends AbstractCollection<Object> implements Output<Object> {

  private Object[] labels;

  public ClassifierOutput(Object[] labels) {
    this.labels = labels;
  }


  public Object get(int i) {
    return labels[i];
  }

  @Override
  public int size() {
    return labels.length;
  }

  @Override
  public Iterator<Object> iterator() {
    return Arrays.asList(labels).iterator();
  }

  @Override
  public Object[] toArray() {
    return labels;
  }
}
