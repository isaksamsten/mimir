package org.mimirframework.classification.conformal;

import java.util.List;

import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.Ensemble;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class BootstrapConformalClassifier extends AbstractConformalClassifier {

  private final DoubleArray calibration;
  private final Nonconformity nonconformity;

  protected BootstrapConformalClassifier(DoubleArray calibration, Nonconformity nonconformity,
      Vector classes) {
    super(classes);
    this.calibration = calibration;
    this.nonconformity = nonconformity;
  }

  @Override
  protected Nonconformity getNonconformity() {
    return nonconformity;
  }

  @Override
  protected DoubleArray getCalibration() {
    return calibration;
  }

  public static class Learner implements Predictor.Learner<BootstrapConformalClassifier> {

    private final Predictor.Learner<? extends Ensemble> learner;
    private final ProbabilityCostFunction costFunction;

    public Learner(Predictor.Learner<? extends Ensemble> learner,
        ProbabilityCostFunction costFunction) {
      this.learner = learner;
      this.costFunction = costFunction;
    }

    @Override
    public BootstrapConformalClassifier fit(DataFrame x, Vector y) {
      Ensemble ensemble = learner.fit(x, y);

      // m x n, m = examples, n = models
      BooleanArray oob = ensemble.getOobIndicator();
      List<Classifier> members = ensemble.getEnsembleMembers();
      DoubleArray nonConformity = DoubleArray.zeros(x.rows());
      for (int i = 0; i < oob.rows(); i++) {
        Vector e = x.loc().getRecord(i);
        BooleanArray o = oob.getRow(i);
        DoubleArray estimate = estimate(members, o, e, ensemble.getClasses());
        int trueClassIndex = ensemble.getClasses().loc().indexOf(y.loc().get(i));
        nonConformity.set(i, costFunction.apply(estimate, trueClassIndex));
      }
      return new BootstrapConformalClassifier(nonConformity,
          new ProbabilityEstimateNonconformity(ensemble, costFunction), ensemble.getClasses());
    }

    private DoubleArray estimate(List<Classifier> members, BooleanArray oob, Vector example,
        Vector classes) {
      DoubleArray probabilities = DoubleArray.zeros(classes.size());
      double oobs = 0;
      for (int i = 0; i < members.size(); i++) {
        if (oob.get(i)) {
          DoubleArray p = members.get(i).estimate(example);
          probabilities.plusAssign(p);
          oobs += 1;
        }
      }
      probabilities.divAssign(oobs);
      return probabilities;
    }

  }

}
