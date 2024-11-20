/*
 * Copyright 2015-2022 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_rtc/gui/NumberInput.h>
#include <mc_tasks/MinimumJerkTask.h>

#include <mc_rtc/gui/Point3D.h>

namespace mc_tasks {

MinimumJerkTask::MinimumJerkTask(const std::string &bodyName,
                                 const mc_rbdyn::Robots &robots,
                                 unsigned int robotIndex, double weight,
                                 bool useFilter)
    : MinimumJerkTask(robots.robot(robotIndex).frame(bodyName), weight) {}

MinimumJerkTask::MinimumJerkTask(const mc_rbdyn::RobotFrame &frame,
                                 double weight, bool useFilter)
    : PositionTask(frame, 0.0, weight), frame_(frame),
      bodyName_(frame_->body()), init_(true), gamma_state_(1.0),
      gamma_output_(1.0), qp_state("QP_success"), W_(0.03), max_L_(2.0),
      max_tau_(0.999), lambda_L_(100), lambda_tau_(100), fitts_a_(0),
      fitts_b_(1.0), max_jac_tau_(1.0), lambda_jac_L_(1e-2),
      lambda_jac_tau_(0.1875), lambda_jac_D_(0.01), gain_linear_cost_(1.0),
      filter_out(useFilter),
      jac_(new rbd::Jacobian(frame.robot().mb(), frame.body())), solver_() {
  switch (backend_) {
  case Backend::TVM:
    break;
  default:
    mc_rtc::log::error_and_throw(
        "[MinimumJerkTask] Not implemented for backend: {}", backend_);
  }
  mc_rtc::log::info("[MinimumJerkTask] Checked backend");
  type_ = "min_jerk";
  name_ = "min_jerk_" + frame.robot().name() + "_" + frame.name();

  LQR_Q_.setZero();
  LQR_R_.setZero();
  W_e_.setZero();
  W_u_.setZero();
  K_.setZero();
  P_.setZero();
  Q_.setZero();
  x_.setZero();
  dx_.setZero();
  f_.setZero();
  g_.setZero();
  u_.setZero();
  A_.setZero();
  B_.setZero();
  err_mj_.setZero();
  err_lyap_.setZero();
  K_ev_.setZero();
  D_.setZero();
  commanded_acc_.setZero();
  disturbance_acc_.setZero();

  LQR_Q_.block<3, 3>(0, 0) = 1e6 * Eigen::Matrix3d::Identity();
  LQR_Q_.block<3, 3>(3, 3) = 1e6 * Eigen::Matrix3d::Identity();
  LQR_Q_.block<3, 3>(6, 6) = 1e0 * Eigen::Matrix3d::Identity();
  LQR_R_ = 1e0 * Eigen::Matrix3d::Identity();
  W_e_.block<3, 3>(0, 0) = 1e6 * Eigen::Matrix3d::Identity();
  W_e_.block<3, 3>(3, 3) = 1e6 * Eigen::Matrix3d::Identity();
  W_e_.block<3, 3>(6, 6) = 1e0 * Eigen::Matrix3d::Identity();
  W_u_.block<3, 3>(0, 0) = 1e0 * Eigen::Matrix3d::Identity();
  W_u_(3, 3) = 1e2;
  W_u_(4, 4) = 1e1;
  W_u_.block<3, 3>(5, 5) = 1e1 * Eigen::Matrix3d::Identity();

  A_.setZero();
  B_.setZero();
  K_.setZero();
  K_.block<3, 3>(0, 0) = -1e3 * Eigen::Matrix3d::Identity();
  K_.block<3, 3>(0, 3) = -1044.7 * Eigen::Matrix3d::Identity();
  K_.block<3, 3>(0, 6) = -45.7214 * Eigen::Matrix3d::Identity();
  K_ev_.setZero();
  A_.block<6, 6>(0, 3).setIdentity();
  B_.block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity();
  solver_.problem(8, 0, 1);
  g_.block<5, 5>(6, 0).setIdentity();

  L_ = 0.5;
  x_(9) = L_;
  tau_ = 0.0;
  x_(10) = tau_;

  H_QP_ = 2 * W_u_.transpose() * W_u_;
}

void MinimumJerkTask::computeDuration(void) {
  if ((L_ / W_) < 1.0) {
    mc_rtc::log::error_and_throw<std::range_error>(
        "Issue in trajectory length and width ratio for Fitts's "
        "duration computation. \n Length = {} | Width = {}",
        L_, W_);
  }
  T_fitts_ = fitts_a_ + fitts_b_ * std::log2(2.0 * L_ / W_);

  T_ = T_fitts_;
}

void MinimumJerkTask::computeMinJerkState(void) {
  double tau2 = tau_ * tau_; // tau^2
  double tau3 = tau2 * tau_; // tau^3
  double tau4 = tau3 * tau_; // tau^4
  double tau5 = tau4 * tau_; // tau^5
  err_mj_.block<3, 1>(0, 0) =
      -L_ * (-6.0 * tau5 + 15.0 * tau4 - 10.0 * tau3 + 1.0) * D_;
  err_mj_.block<3, 1>(3, 0) =
      -(L_ / T_) * (-30.0 * tau4 + 60.0 * tau3 - 30.0 * tau2) * D_;
  err_mj_.block<3, 1>(6, 0) =
      -(L_ / pow(T_, 2)) * (-120.0 * tau3 + 180.0 * tau2 - 60.0 * tau_) * D_;
}

void MinimumJerkTask::computeF(void) {
  f_.block<3, 1>(0, 0) = x_.block<3, 1>(3, 0);
  f_.block<3, 1>(3, 0) = x_.block<3, 1>(6, 0);
  f_.block<3, 1>(6, 0) =
      -(L_ / pow(T_, 3)) * (-360.0 * tau_ * tau_ + 360.0 * tau_ - 60.0) * D_;
  f_(10) = 1 / T_;
}

void MinimumJerkTask::updateB(void) {
  double tau = (tau_ > max_jac_tau_) ? max_jac_tau_ : tau_;
  double tau2 = tau * tau;  // tau^2
  double tau3 = tau2 * tau; // tau^3
  double tau4 = tau3 * tau; // tau^4
  double tau5 = tau4 * tau; // tau^5
  double ln2 = std::log(2);
  Eigen::Matrix3d SD;
  SD << 0, -D_(2), D_(1), D_(2), 0, -D_(0), -D_(1), D_(0), 0;

  // Jacobian for position error
  B_.block<3, 1>(0, 3) =
      (6.0 * tau5 - 15.0 * tau4 + 10.0 * tau3 - 1.0) * D_ - lambda_jac_L_ * D_;
  B_.block<3, 1>(0, 4) = L_ * (30.0 * tau4 - 60.0 * tau3 + 30.0 * tau2) * D_ +
                         L_ * lambda_jac_tau_ * D_;
  B_.block<3, 3>(0, 5) =
      L_ * (6.0 * tau5 - 15.0 * tau4 + 10.0 * tau3 - 1.0) * SD -
      lambda_jac_D_ * SD;

  // Jacobian for velocity error
  B_.block<3, 1>(3, 3) = -((30 * tau4 - 60 * tau3 + 30 * tau2) *
                           (fitts_b_ - ln2 * T_) / (ln2 * pow(T_, 2))) *
                         D_;
  B_.block<3, 1>(3, 4) =
      (L_ / T_) * (120.0 * tau3 - 180.0 * tau2 + 60.0 * tau) * D_;
  B_.block<3, 3>(3, 5) =
      (L_ / T_) * (30.0 * tau4 - 60.0 * tau3 + 30.0 * tau2) * SD;

  // Jacobian for acceleration error
  B_.block<3, 1>(6, 3) = -((120.0 * tau3 - 180.0 * tau2 + 60.0 * tau) *
                           (2 * fitts_b_ - ln2 * T_) / (ln2 * pow(T_, 3))) *
                         D_;
  B_.block<3, 1>(6, 4) =
      (L_ / pow(T_, 2)) * (360.0 * tau * tau - 360.0 * tau + 60.0) * D_;
  B_.block<3, 3>(6, 5) =
      (L_ / pow(T_, 2)) * (120.0 * tau3 - 180.0 * tau2 + 60.0 * tau) * SD;
}

void MinimumJerkTask::updateG(void) {
  Eigen::Matrix3d SD;
  SD << 0, -D_(2), D_(1), D_(2), 0, -D_(0), -D_(1), D_(0), 0;
  g_.block<3, 3>(11, 5) = SD;
}

void MinimumJerkTask::solveLQR(void) {
  auto A = A_;
  auto B = B_.block<9, 3>(0, 0);
  auto R = LQR_R_;
  auto Q = LQR_Q_;

  size_t n = static_cast<size_t>(A.rows());

  // Form the Hamiltoninan matrix Z
  Eigen::MatrixXd Z(2 * n, 2 * n);
  Z << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();
  mc_rtc::log::info("Z matrix: {}", Z);
  std::vector<int> stable_indices;
  Eigen::MatrixXcd eigenvalues;
  Eigen::MatrixXcd eigenvectors;

  for (int k = 0; k < 10; k++) {
    // Compute eigenvalues and eigenvectors
    Eigen::ComplexEigenSolver<Eigen::MatrixXd> eig_solver(Z);
    eigenvalues = eig_solver.eigenvalues();
    eigenvectors = eig_solver.eigenvectors();

    // Print eigenvalues for debugging
    std::cout << "Eigenvalues of Z:\n" << eigenvalues << "\n\n";

    // Select the stable eigenvectors (corresponding to eigenvalues with
    // negative real parts)
    stable_indices.clear();
    for (size_t i = 0; i < 2 * n; ++i) {
      if (eigenvalues(int(i)).real() < 0) {
        stable_indices.push_back(int(i));
      }
    }
    if (stable_indices.size() == n) {
      break;
    }
  }

  // Check if we found exactly 'n' stable eigenvalues
  if (stable_indices.size() != n) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "Could not find exactly 'n' stable eigenvalues.");
  }

  // Sort stable indices by the real part of eigenvalues (descending)
  std::sort(stable_indices.begin(), stable_indices.end(), [&](int a, int b) {
    return eigenvalues(a).real() > eigenvalues(b).real();
  });

  // Form the stable subspace using the stable eigenvectors
  Eigen::MatrixXcd V_stable(2 * n, n);
  for (size_t i = 0; i < n; ++i) {
    std::cout << "Selected eigenvalue: " << eigenvalues(stable_indices[i])
              << std::endl;
    std::cout << "Associated eigenvector: "
              << eigenvectors.col(stable_indices[i]).head(int(n)) << std::endl;
    V_stable.col(int(i)) = eigenvectors.col(stable_indices[i]);
  }

  // Partition the eigenvector matrix into V1 and V2
  Eigen::MatrixXcd V1 = V_stable.topRows(int(n));    // Top half (V1)
  Eigen::MatrixXcd V2 = V_stable.bottomRows(int(n)); // Bottom half (V2)

  // Compute the Riccati solution X = V2 * V1.inverse()
  Eigen::MatrixXd P = (V2 * V1.inverse()).real();
  Eigen::IOFormat format(5, 0, " ", "\n", "[", "]", "", "");

  Eigen::MatrixXd dPdt(n, n);
  dPdt.setOnes();
  for (int i = 0; i < 10000; i++) {
    dPdt = P * A + A.transpose() * P -
           (P * B) * R.inverse() * (B.transpose() * P) + Q;
    P = P + 0.01 * dPdt;
    if (dPdt.norm() < 1e-8) {
      break;
    }
  }

  // Check validity of Riccati solution
  Eigen::MatrixXd R_inv = R.inverse();
  Eigen::MatrixXd left_side =
      A_.transpose() * P + P * A_ - P * B * R_inv * B.transpose() * P + Q;

  // Check lyapunov
  Eigen::MatrixXd Q_lyap = (R.inverse() * B.transpose() * P).transpose() * R *
                               (R.inverse() * B.transpose() * P) +
                           LQR_Q_;
  Eigen::MatrixXd lyap =
      Q_lyap + P * (A - B * (R.inverse() * B.transpose() * P)) +
      (A - B * (R.inverse() * B.transpose() * P)).transpose() * P;
  if (left_side.isZero(1e-6) and lyap.isZero(1e-6)) {
    // We can take the new solution
    P_ = P;
    K_ = R.inverse() * B.transpose() * P_;
    mc_rtc::log::info("[MinimumJerkTask] Solved LQR");
    Eigen::IOFormat format(5, 0, " ", "\n", "[", "]", "", "");
    std::cout << "P matrix\n" << P_.format(format) << "\n";
    std::cout << "K matrix\n" << K_.format(format) << "\n";
    computeQ();
  } else {
    auto K = R.inverse() * B.transpose() * P;
    mc_rtc::log::warning("[MinimumJerkTask] Failed to solve LQR, computed "
                         "solution does not verifies Riccati equation");
    Eigen::IOFormat format(5, 0, " ", "\n", "[", "]", "", "");
    std::cout << "Left side: \n" << left_side.format(format) << "\n";
    std::cout << "P matrix\n" << P.format(format) << "\n";
    std::cout << "K matrix\n" << K.format(format) << "\n";
  }
}

void MinimumJerkTask::update(mc_solver::QPSolver &solver) {
  auto &robot = frame_->robot();
  mc_tvm::Robot &tvm_robot = robot.tvmRobot();

  auto J = jac_->jacobian(robot.mb(), robot.mbc());
  disturbance_acc_ = J * tvm_robot.alphaDExternal();

  sva::PTransform transform(robot.bodyPosW(bodyName_));
  curr_pos_ = robot.bodyPosW(bodyName_).translation();
  Eigen::Vector3d vel = robot.bodyVelW(bodyName_).linear();
  Eigen::Vector3d acc =
      transform.rotation().transpose() * robot.bodyAccB(bodyName_).linear() +
      robot.bodyVelW(bodyName_).angular().cross(vel) -
      gamma_state_ * disturbance_acc_.tail<3>();
  // Eigen::Vector3d acc =
  //     transform.rotation().transpose() * robot.bodyAccB(bodyName_).linear();
  //     NOT CORRECT

  Eigen::Vector3d err = target_pos_ - curr_pos_;
  if (init_) {
    D_ << -err / err.norm();
    x_.block<3, 1>(11, 0) = D_;
    L_ = std::max(err.norm(), W_);
    x_(9) = L_;
    tau_ = 0.0;
    x_(10) = tau_;
    reaction_time_counter_ = 0;
    commanded_acc_.setZero();
    init_ = false;
  }

  x_.block<3, 1>(0, 0) = err;
  x_.block<3, 1>(3, 0) = -vel;
  x_.block<3, 1>(6, 0) = -acc;

  L_ = x_(9);
  tau_ = x_(10);
  D_ = x_.tail(3);

  if (reaction_time_counter_ < reaction_time_) {
    reaction_time_counter_ += solver.dt();
    PositionTask::refAccel(Eigen::Vector3d::Zero());
    PositionTask::update(solver);
    return;
  }

  computeDuration();
  computeMinJerkState();
  computeF();
  auto prev_err_lyap = err_lyap_;
  err_lyap_ = err_mj_ - x_.head(9);
  mj_pose_ = target_pos_ - err_mj_.head(3);

  K_ev_.block<3, 1>(0, 0) = K_ * err_lyap_;
  Eigen::IOFormat format(5, 0, " ", "\n", "[", "]", "", "");
  dev_diff_ = (err_lyap_ - prev_err_lyap) / solver.dt();
  dev_pred_ = A_ * prev_err_lyap + B_ * u_;
  dyn_error_ = (dev_diff_ - dev_pred_);

  updateB();
  updateG();

  // Compute the matrices to put in the QP
  f_QP_ = gain_linear_cost_ * err_lyap_.transpose() * W_e_ * P_ * B_;
  A_QP_ = err_lyap_.transpose() * P_ * B_;
  b_QP_ = -err_lyap_.transpose() * P_ * B_ * K_ev_;
  lb_QP_ << -100, -100, -100, lambda_L_ * (W_ - L_),
      -lambda_tau_ * tau_ - (1 / T_), -100, -100, -100;
  ub_QP_ << 100, 100, 100, lambda_L_ * (max_L_ - L_),
      lambda_tau_ * (max_tau_ - tau_) - (1 / T_), 100, 100, 100;

  // Solve QP
  bool success = solver_.solve(H_QP_, f_QP_, Eigen::MatrixXd::Zero(0, 0),
                               Eigen::VectorXd::Zero(0), A_QP_, b_QP_, lb_QP_,
                               ub_QP_, false, 1e-6);

  // Handle QP fails
  if (not success) {
    // mc_rtc::log::warning(
    //     "MinimumJerkTask's QP failed,applying worst case convergence");
    u_ = -K_ev_;
    qp_state = "QP_failed";
  } else {
    u_ = solver_.result();
    qp_state = "QP_success";
  }

  dx_ = f_ + g_ * u_;
  dx_(10) = (x_(10) + dx_(10) * solver.dt() < max_tau_)
                ? dx_(10)
                : (max_tau_ - x_(10)) / solver.dt();
  dx_(9) =
      (x_(9) + dx_(9) * solver.dt() > W_) ? dx_(9) : (W_ - x_(9)) / solver.dt();
  x_ = x_ + dx_ * solver.dt();
  x_.tail<3>().normalize();
  commanded_acc_ = commanded_acc_ + dx_.block<3, 1>(6, 0) * solver.dt();
  ref_acc_ = -commanded_acc_ + gamma_output_ * disturbance_acc_.tail<3>();

  // Set PositionTask's refAccel
  PositionTask::refAccel(ref_acc_);

  PositionTask::update(solver);
}

void MinimumJerkTask::addToLogger(mc_rtc::Logger &logger) {
  // ========== State logging ========== //
  logger.addLogEntry("MinimumJerkTask_state_error_pos", this,
                     [this]() -> Eigen::Vector3d { return x_.head<3>(); });
  logger.addLogEntry(
      "MinimumJerkTask_state_error_vel", this,
      [this]() -> Eigen::Vector3d { return x_.block<3, 1>(3, 0); });
  logger.addLogEntry(
      "MinimumJerkTask_state_error_acc", this,
      [this]() -> Eigen::Vector3d { return x_.block<3, 1>(6, 0); });
  logger.addLogEntry("MinimumJerkTask_state_traj_length", this,
                     [this]() -> double { return x_(9); });
  logger.addLogEntry("MinimumJerkTask_state_traj_phase", this,
                     [this]() -> double { return x_(10); });
  logger.addLogEntry("MinimumJerkTask_state_traj_direction", this,
                     [this]() -> Eigen::Vector3d { return x_.tail<3>(); });

  // ========== State derivative logging ========== //
  logger.addLogEntry("MinimumJerkTask_stateD_error_vel", this,
                     [this]() -> Eigen::Vector3d { return dx_.head<3>(); });
  logger.addLogEntry(
      "MinimumJerkTask_stateD_error_acc", this,
      [this]() -> Eigen::Vector3d { return dx_.block<3, 1>(3, 0); });
  logger.addLogEntry(
      "MinimumJerkTask_stateD_error_jerk", this,
      [this]() -> Eigen::Vector3d { return dx_.block<3, 1>(6, 0); });
  logger.addLogEntry("MinimumJerkTask_stateD_traj_length", this,
                     [this]() -> double { return dx_(9); });
  logger.addLogEntry("MinimumJerkTask_stateD_traj_phase", this,
                     [this]() -> double { return dx_(10); });
  logger.addLogEntry("MinimumJerkTask_stateD_traj_direction", this,
                     [this]() -> Eigen::Vector3d { return dx_.tail<3>(); });

  // ========== Minimum Jerk traj related ========== //
  logger.addLogEntry("MinimumJerkTask_minimum_jerk_state_error", this,
                     [this]() -> Eigen::Vector3d { return err_mj_.head<3>(); });
  logger.addLogEntry(
      "MinimumJerkTask_minimum_jerk_state_vel", this,
      [this]() -> Eigen::Vector3d { return err_mj_.block<3, 1>(3, 0); });
  logger.addLogEntry("MinimumJerkTask_minimum_jerk_state_acc", this,
                     [this]() -> Eigen::Vector3d { return err_mj_.tail<3>(); });
  logger.addLogEntry("MinimumJerkTask_minimum_jerk_duration", this,
                     [this]() { return T_; });

  // ========== Fitts related ========== //
  logger.addLogEntry("MinimumJerkTask_Fitts_a", this,
                     [this]() { return fitts_a_; });
  logger.addLogEntry("MinimumJerkTask_Fitts_b", this,
                     [this]() { return fitts_b_; });
  logger.addLogEntry("MinimumJerkTask_Fitts_W", this, [this]() { return W_; });
  logger.addLogEntry("MinimumJerkTask_Fitts_duration", this,
                     [this]() { return T_fitts_; });
  logger.addLogEntry("MinimumJerkTask_Fitts_reaction_time", this,
                     [this]() { return reaction_time_; });

  // ========== QP related ========== //
  logger.addLogEntry("MinimumJerkTask_QP_state", this,
                     [this]() { return qp_state; });
  logger.addLogEntry("MinimumJerkTask_QP_state_cost_vec", this,
                     [this]() -> Eigen::VectorXd { return f_QP_; });
  logger.addLogEntry("MinimumJerkTask_QP_output_cost_vec", this,
                     [this]() -> Eigen::VectorXd { return H_QP_.diagonal(); });
  logger.addLogEntry("MinimumJerkTask_QP_state_cost", this,
                     [this]() -> double { return f_QP_.transpose() * u_; });
  logger.addLogEntry(
      "MinimumJerkTask_QP_output_cost", this,
      [this]() -> double { return 0.5 * u_.transpose() * H_QP_ * u_; });
  logger.addLogEntry("MinimumJerkTask_QP_cost", this, [this]() -> double {
    return 0.5 * static_cast<double>(u_.transpose() * H_QP_ * u_) +
           static_cast<double>(f_QP_.transpose() * u_);
  });
  logger.addLogEntry("MinimumJerkTask_QP_W_state", this,
                     [this]() { return W_e(); });
  logger.addLogEntry("MinimumJerkTask_QP_W_output", this,
                     [this]() { return W_u(); });

  // ========== Other ========== //
  logger.addLogEntry("MinimumJerkTask_target_pose", this,
                     [this]() { return target_pos_; });
  logger.addLogEntry("MinimumJerkTask_control_signal", this,
                     [this]() { return u_; });
  logger.addLogEntry("MinimumJerkTask_err", this, [this]() { return eval(); });
  logger.addLogEntry("MinimumJerkTask_err_norm", this,
                     [this]() { return eval().norm(); });
  logger.addLogEntry("MinimumJerkTask_commanded_acc", this,
                     [this]() -> Eigen::Vector3d { return commanded_acc_; });
  logger.addLogEntry(
      "MinimumJerkTask_disturb_acc", this,
      [this]() -> Eigen::Vector3d { return disturbance_acc_.tail<3>(); });
  logger.addLogEntry("MinimumJerkTask_ref_acc", this,
                     [this]() -> Eigen::Vector3d { return ref_acc_; });
  logger.addLogEntry("MinimumJerkTask_lyap_error", this,
                     [this]() -> Eigen::VectorXd { return err_lyap_; });
  logger.addLogEntry("MinimumJerkTask_lyap_pred_dyn", this,
                     [this]() -> Eigen::VectorXd { return dev_pred_; });
  logger.addLogEntry("MinimumJerkTask_lyap_diff_dyn", this,
                     [this]() -> Eigen::VectorXd { return dev_diff_; });
  logger.addLogEntry("MinimumJerkTask_lyap_dyn_error_vec", this,
                     [this]() -> Eigen::VectorXd { return dyn_error_; });
  logger.addLogEntry("MinimumJerkTask_lyap_dyn_error_norm", this,
                     [this]() { return dyn_error_.norm(); });
  // PositionTask::addToLogger(logger);
}

void MinimumJerkTask::addToGUI(mc_rtc::gui::StateBuilder &gui) {
  gui.addElement(
      {"Tasks", name_, "Hyperparameters", "Controller"},
      mc_rtc::gui::NumberInput("Max tau", max_tau_),
      mc_rtc::gui::NumberInput("Jacobian's max tau", max_jac_tau_),
      mc_rtc::gui::NumberInput("Jacobians damping on L", lambda_jac_L_),
      mc_rtc::gui::NumberInput("Jacobians damping on tau", lambda_jac_tau_),
      mc_rtc::gui::NumberInput("Jacobians damping on D", lambda_jac_D_),
      mc_rtc::gui::NumberInput("Linear cost gain [0;1]", gain_linear_cost_),
      mc_rtc::gui::NumberInput("Input gamma", gamma_state_),
      mc_rtc::gui::NumberInput("Output gamma", gamma_output_));
  gui.addElement({"Tasks", name_, "Hyperparameters", "Fitts"},
                 mc_rtc::gui::NumberInput("Fitts's constant (a)", fitts_a_),
                 mc_rtc::gui::NumberInput("Fitts's proportional (b)", fitts_b_),
                 mc_rtc::gui::NumberInput("Reaction time", reaction_time_));
  gui.addElement({"Tasks", name_, "Gains"},
                 mc_rtc::gui::ArrayInput(
                     "LQR weight", {"e", "v", "a", "j"},
                     [this]() -> Eigen::Vector4d {
                       auto wq = LQR_Q();
                       return Eigen::Vector4d(wq(0), wq(1), wq(2), LQR_R());
                     },
                     [this](Eigen::Vector4d v) {
                       LQR_Q(v.head(3));
                       LQR_R(v(3));
                     }),
                 mc_rtc::gui::ArrayInput(
                     "QP state weight", {"e", "v", "a"},
                     [this]() -> Eigen::Vector3d { return W_e(); },
                     [this](Eigen::Vector3d v) { W_e(v); }),
                 mc_rtc::gui::ArrayInput(
                     "QP output weight", {"J", "L", "tau", "D"},
                     [this]() -> Eigen::Vector4d { return W_u(); },
                     [this](Eigen::Vector4d v) { W_u(v); }),
                 mc_rtc::gui::ArrayLabel(
                     "Lyapunov error",
                     {"ex", "ey", "ez", "vx", "vy", "vz", "ax", "ay", "az"},
                     [this]() { return err_lyap_; }),
                 mc_rtc::gui::ArrayLabel(
                     "QP output",
                     {"jx", "jy", "jz", "length", "phase", "Dx", "Dy", "Dz"},
                     [this]() { return u_; }));
  gui.addElement(
      {"Tasks", name_, "Computed"},
      mc_rtc::gui::Label("Trajectory length", [this]() { return L_; }),
      mc_rtc::gui::Label("Trajectory phase", [this]() { return tau_; }),
      mc_rtc::gui::Label("Trajectory duration", [this]() { return T_; }),
      mc_rtc::gui::Label("Fitts duration", [this]() { return T_fitts_; }),
      mc_rtc::gui::Label("Target size", [this]() { return W_; }),
      mc_rtc::gui::ArrayLabel(
          "Lyapunov error",
          {"ex", "ey", "ez", "vx", "vy", "vz", "ax", "ay", "az"},
          [this]() { return err_lyap_; }),
      mc_rtc::gui::ArrayLabel(
          "QP output", {"jx", "jy", "jz", "length", "phase", "Dx", "Dy", "Dz"},
          [this]() { return u_; }),
      mc_rtc::gui::ArrayLabel(
          "Robot error", {"ex", "ey", "ez", "vx", "vy", "vz", "ax", "ay", "az"},
          [this]() -> Eigen::VectorXd { return x_.head<9>(); }),
      mc_rtc::gui::ArrayLabel(
          "MJ error", {"ex", "ey", "ez", "vx", "vy", "vz", "ax", "ay", "az"},
          [this]() { return err_mj_; }),
      mc_rtc::gui::ArrayLabel("MjState", {"ex", "ey", "ez"},
                              [this]() { return mj_pose_; }));
  gui.addElement(
      {"Tasks", name_, "Visual"},
      mc_rtc::gui::Arrow(
          "MJ Direction",
          mc_rtc::gui::ArrowConfig(mc_rtc::gui::Color(0.0, 0.0, 1.0, 0.3)),
          [this]() -> Eigen::Vector3d { return target_pos_; },
          [this]() -> Eigen::Vector3d { return target_pos_ + 0.1 * D_; }),
      mc_rtc::gui::Arrow(
          "Acc Direction",
          mc_rtc::gui::ArrowConfig(mc_rtc::gui::Color(0.0, 1.0, 0.0, 0.3)),
          [this]() -> Eigen::Vector3d { return curr_pos_; },
          [this]() -> Eigen::Vector3d {
            return curr_pos_ - 0.1 * x_.block<3, 1>(6, 0).normalized();
          }),
      mc_rtc::gui::Point3DRO("Target position", target_pos_),
      mc_rtc::gui::Point3DRO(
          "MinJerk state position",
          mc_rtc::gui::PointConfig(mc_rtc::gui::Color(0.0, 1.0, 0.0)),
          mj_pose_));
  // PositionTask::addToGUI(gui);
}
} // namespace mc_tasks
