package org.mimirframework.classification;

import org.briljantframework.data.vector.Vector;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierMeasureTest {

  @Test
  public void testGetPrecision() throws Exception {
    Vector t = Vector.of(1, 1, 1, 2);
    Vector p = Vector.of(1, 1, 2, 1);
    ClassifierMeasure cm = new ClassifierMeasure(t, p);

    assertEquals(0.33, cm.getPrecision(), 0.01);
    assertEquals(0.33, cm.getRecall(), 0.01);
    assertEquals(0.5, cm.getAccuracy(), 0.01);
  }
}
