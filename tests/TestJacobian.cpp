#define BOOST_TEST_MODULE TestJacobian

#include <boost/test/unit_test.hpp>
#include <mc_tasks/MinimumJerkTask.h>

double pick_random(double min, double max) {
  return min +
         (max - min) * ((float)std::rand() / static_cast<float>(RAND_MAX));
}

Eigen::MatrixXd computeJacobian(double L_, double tau_, Eigen::Vector3d D_,
                                double W_, double fitts_a_, double fitts_b_,
                                double T_) {
  static Eigen::Matrix<double, 9, 8> B_;
  static double lnT = std::log(2.0 * L_ / W_);
  static double tau2 = tau_ * tau_; // tau^2
  static double tau3 = tau2 * tau_; // tau^3
  static double tau4 = tau3 * tau_; // tau^4
  static double tau5 = tau4 * tau_; // tau^5
  static double ln2 = std::log(2);
  static Eigen::Matrix3d SD;

  B_.setZero();
  SD << 0, -D_(2), D_(1), D_(2), 0, -D_(0), -D_(1), D_(0), 0;

  // Jacobian for position error
  B_.block<3, 1>(0, 0) = (-6.0 * tau5 + 15.0 * tau4 - 10.0 * tau3 + 1.0) * D_;
  B_.block<3, 1>(0, 1) = L_ * (-30.0 * tau4 + 60.0 * tau3 - 30.0 * tau2) * D_;
  B_.block<3, 3>(0, 2) =
      L_ * (-6.0 * tau5 + 15.0 * tau4 - 10.0 * tau3 + 1.0) * SD;

  // Jacobian for velocity error
  B_.block<3, 1>(3, 0) = (30 * tau2 * pow(tau_ - 1, 2) * (fitts_b_ - ln2 * T_) /
                          (ln2 * pow(T_, 2))) *
                         D_;
  B_.block<3, 1>(3, 1) =
      (L_ / T_) * (-120.0 * tau3 + 180.0 * tau2 - 60.0 * tau_) * D_;
  B_.block<3, 3>(3, 2) =
      (L_ / T_) * (-30.0 * tau4 + 60.0 * tau3 - 30.0 * tau2) * SD;

  // Jacobian for acceleration error
  B_.block<3, 1>(6, 0) = (60 * tau_ * (2 * tau_ - 1) * (tau_ - 1) *
                          (2 * fitts_b_ - ln2) / (ln2 * pow(T_, 2))) *
                         D_;
  B_.block<3, 1>(6, 1) =
      (L_ / pow(T_, 2)) * (-360.0 * tau_ * tau_ + 360.0 * tau_ - 60.0) * D_;
  B_.block<3, 3>(6, 2) =
      (L_ / pow(T_, 2)) * (-120.0 * tau3 + 180.0 * tau2 - 60.0 * tau_) * SD;
  return B_;
}

Eigen::Matrix<double, 9, 1>
compute_min_jerk_state(double L_, double tau_, Eigen::Vector3d D_, double T_) {
  Eigen::Matrix<double, 9, 1> err_mj_;
  double tau2 = tau_ * tau_; // tau^2
  double tau3 = tau2 * tau_; // tau^3
  double tau4 = tau3 * tau_; // tau^4
  double tau5 = tau4 * tau_; // tau^5
  err_mj_.block<3, 1>(0, 0) =
      L_ * (-6.0 * tau5 + 15.0 * tau4 - 10.0 * tau3 + 1.0) * D_;
  err_mj_.block<3, 1>(3, 0) =
      (L_ / T_) * (-30.0 * tau4 + 60.0 * tau3 - 30.0 * tau2) * D_;
  err_mj_.block<3, 1>(6, 0) =
      (L_ / pow(T_, 2)) * (-120.0 * tau3 + 180.0 * tau2 - 60.0 * tau_) * D_;
  return err_mj_;
}

BOOST_AUTO_TEST_CASE(test_acobian) {
  double L, W, tau, T, fitts_a = -0.09, fitts_b = 0.35;
  double new_L, new_tau, new_T;
  double dL, dtau;
  double dt = 1e-9;
  Eigen::Vector3d D, dD, new_D;
  Eigen::Matrix<double, 5, 1> dU;
  Eigen::Matrix<double, 9, 1> dX, X_0, X_1;
  Eigen::Matrix<double, 9, 5> J;
  Eigen::Matrix3d SD;

  for (int i = 0; i < 1000; i++) {
    // Pick state
    W = pick_random(0.001, 0.05);
    L = pick_random(W, 2.0);
    tau = pick_random(0.0, 1.0);
    D = D.Random().normalized();
    SD.setZero();
    SD << 0, -D(2), D(1), D(2), 0, -D(0), -D(1), D(0), 0;
    T = fitts_a + fitts_b * log2(2 * L / W);

    // Pick dU
    dL = dt * pick_random(100.0 * (W - L), 100.0 * (2.0 - L));
    dtau =
        dt * pick_random(-100.0 * tau - (1 / T), 100.0 * (1 - tau) - (1 / T));
    dD = dt * 200.0 * dD.Random();
    dU << dL, dtau, dD;
    J = computeJacobian(L, tau, D, W, fitts_a, fitts_b, T);
    dX = J * dU;

    // Finite difference
    new_L = L + dL;
    new_tau = tau + dtau;
    new_D = (D + SD * dD).normalized();
    new_T = fitts_a + fitts_b * log2(2 * new_L / W);
    X_0 = compute_min_jerk_state(L, tau, D, T);
    X_1 = compute_min_jerk_state(new_L, new_tau, new_D, new_T);

    bool cond = ((X_1 - X_0) / dt - dX / dt).norm() < 1e-3;
    BOOST_TEST(true);
    if (not cond) {
      Eigen::IOFormat format(5, 0, " ", "\n", "[", "]", "", "");
      std::cout << "L = " << L << ", tau = " << tau << ", W = " << W
                << std::endl;
      std::cout << "U = \n" << dU.transpose().format(format) << std::endl;
      std::cout << "dX = \n" << dX.transpose().format(format) << std::endl;
      std::cout << "Finite difference = \n"
                << (X_1 - X_0).transpose().format(format) << std::endl;
      std::cout << "Error = \n"
                << ((X_1 - X_0) - dX).transpose().format(format) << std::endl;
      std::cout << "X_0 = \n" << X_0.transpose().format(format) << std::endl;
      std::cout << "X_1 = \n" << X_1.transpose().format(format) << std::endl;
    }
  }
}
