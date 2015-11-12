package org.mimirframework.classification;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public enum EnsembleMeasure {
  BASE_ERROR, OOB_ERROR, BIAS, VARIANCE, MSE, QUALITY, CORRELATION, STRENGTH, ERROR_BOUND;
}
