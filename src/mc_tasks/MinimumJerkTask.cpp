/*
 * Copyright 2015-2022 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_tasks/MinimumJerkTask.h>

#include <mc_rtc/gui/Point3D.h>

namespace mc_tasks {

MinimumJerkTask::MinimumJerkTask(const std::string &bodyName,
                                 const mc_rbdyn::Robots &robots,
                                 unsigned int robotIndex, double weight)
    : MinimumJerkTask(robots.robot(robotIndex).frame(bodyName), weight) {}

MinimumJerkTask::MinimumJerkTask(const mc_rbdyn::RobotFrame &frame,
                                 double weight)
    : PositionTask(frame, 0.0, weight), frame_(frame),
      bodyName_(frame_->body()), init_(true), W_(0.03), max_L_(2.0),
      lambda_L_(100), lambda_tau_(100), fitts_a_(0), fitts_b_(1.0),
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

  W_1_.setZero();
  W_2_.setZero();
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

  W_1_.block<3, 3>(0, 0) = 1e6 * Eigen::Matrix3d::Identity();
  W_1_.block<3, 3>(3, 3) = 1e6 * Eigen::Matrix3d::Identity();
  W_1_.block<3, 3>(6, 6) = 1e0 * Eigen::Matrix3d::Identity();
  W_2_.block<3, 3>(0, 0) = 1e0 * Eigen::Matrix3d::Identity();
  W_2_(3, 3) = 1e2;
  W_2_(4, 4) = 1e1;
  W_2_.block<3, 3>(5, 5) = 1e1 * Eigen::Matrix3d::Identity();

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

  H_QP_ = 2 * W_2_.transpose() * W_2_;
  solveLQR();
}

void MinimumJerkTask::computeFitts(void) {
  if ((L_ / W_) < 1.0) {
    mc_rtc::log::error_and_throw<std::range_error>(
        "Issue in trajectory length and width ratio for Fitts's "
        "duration computation. \n Length = {} | Width = {}",
        L_, W_);
  }
  T_ = fitts_a_ + fitts_b_ * std::log2(2.0 * L_ / W_);
}

void MinimumJerkTask::computeMinJerkState(void) {
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
}

void MinimumJerkTask::computeF(void) {
  f_.block<3, 1>(0, 0) = x_.block<3, 1>(3, 0);
  f_.block<3, 1>(3, 0) = x_.block<3, 1>(6, 0);
  f_.block<3, 1>(6, 0) =
      (L_ / pow(T_, 3)) * (-360.0 * tau_ * tau_ + 360.0 * tau_ - 60.0) * D_;
  f_(10) = 1 / T_;
}

void MinimumJerkTask::updateB(void) {
  double lnT = std::log(2.0 * L_ / W_);
  double tau2 = tau_ * tau_; // tau^2
  double tau3 = tau2 * tau_; // tau^3
  double tau4 = tau3 * tau_; // tau^4
  double tau5 = tau4 * tau_; // tau^5
  double ln2 = std::log(2);
  Eigen::Matrix3d SD;
  SD << 0, -D_(2), D_(1), D_(2), 0, -D_(0), -D_(1), D_(0), 0;

  // Jacobian for velocity error
  B_.block<3, 1>(0, 3) = (-6.0 * tau5 + 15.0 * tau4 - 10.0 * tau3 + 1.0) * D_;
  B_.block<3, 1>(0, 4) = L_ * (-30.0 * tau4 + 60.0 * tau3 - 30.0 * tau2) * D_;
  B_.block<3, 3>(0, 5) =
      L_ * (-6.0 * tau5 + 15.0 * tau4 - 10.0 * tau3 + 1.0) * SD;

  // Jacobian for acceleration error
  B_.block<3, 1>(3, 3) =
      (-(30.0 * tau2 * ln2 * (lnT - 1) * pow(tau_ - 1.0, 2)) / pow(lnT, 2)) *
      D_;
  B_.block<3, 1>(3, 4) =
      (L_ / T_) * (-120.0 * tau3 + 180.0 * tau2 - 60.0 * tau_) * D_;
  B_.block<3, 3>(3, 5) =
      (L_ / T_) * (-30.0 * tau4 + 60.0 * tau3 - 30.0 * tau2) * SD;

  // Jacobian for jerk error
  B_.block<3, 1>(6, 3) =
      (-(60 * tau_ * pow(ln2, 2) * (lnT - 2) * (2 * tau_ - 1) * (tau_ - 1)) /
       pow(lnT, 3)) *
      D_;
  B_.block<3, 1>(6, 4) =
      (L_ / pow(T_, 2)) * (-360.0 * tau_ * tau_ + 360.0 * tau_ - 60.0) * D_;
  B_.block<3, 3>(6, 5) =
      (L_ / pow(T_, 2)) * (-120.0 * tau3 + 180.0 * tau2 - 60.0 * tau_) * SD;
}

void MinimumJerkTask::updateG(void) {
  Eigen::Matrix3d SD;
  SD << 0, -D_(2), D_(1), D_(2), 0, -D_(0), -D_(1), D_(0), 0;
  g_.block<3, 3>(11, 5) = SD;
}

void MinimumJerkTask::solveLQR(void) {
  auto B = B_.block<9, 3>(0, 0);
  auto R = W_2_.block<3, 3>(0, 0);

  P_.setZero();
  Eigen::Matrix<double, 9, 9> dPdt;
  dPdt.setOnes();
  int iter = 0;
  do {
    dPdt = P_ * A_ + A_.transpose() * P_ -
           (P_ * B) * R.inverse() * (B.transpose() * P_) + W_1_;
    P_ = P_ + 0.01 * dPdt;
    iter += 1;
  } while (dPdt.norm() > 1e-8 and iter < 50000);
  if (dPdt.norm() > 1e-8) {
    mc_rtc::log::error("[MinimumJerkTask] LQR solution didn't converge");
  } else {
    K_ = R.inverse() * B.transpose() * P_;
    mc_rtc::log::info("[MinimumJerkTask] Solved LQR | norm");
    Eigen::IOFormat format(5, 0, " ", "\n", "[", "]", "", "");
    std::cout << "P matrix\n" << P_.format(format) << "\n";
    std::cout << "K matrix\n" << K_.format(format) << "\n";
  }
}

void MinimumJerkTask::update(mc_solver::QPSolver &solver) {
  auto &robot = frame_->robot();
  mc_tvm::Robot &tvm_robot = robot.tvmRobot();
  sva::PTransform transform(robot.bodyPosW(bodyName_));
  curr_pos_ = robot.bodyPosW(bodyName_).translation();
  Eigen::Vector3d vel = robot.bodyVelW(bodyName_).linear();
  Eigen::Vector3d acc =
      transform.rotation().transpose() * robot.bodyAccB(bodyName_).linear() +
      robot.bodyVelW(bodyName_).angular().cross(vel);
  // Eigen::Vector3d acc =
  //     transform.rotation().transpose() * robot.bodyAccB(bodyName_).linear();
  //     NOT CORRECT

  Eigen::Vector3d err = target_pos_ - curr_pos_;
  if (init_) {
    D_ = err / err.norm();
    x_.block<3, 1>(11, 0) = D_;
    L_ = err.norm();
    x_(9) = L_;
    tau_ = 0.0;
    x_(10) = tau_;
    init_ = false;
  }
  x_.block<3, 1>(0, 0) = err;
  x_.block<3, 1>(3, 0) = -vel;
  x_.block<3, 1>(6, 0) = -acc;

  L_ = x_(9);
  tau_ = x_(10);
  D_ = x_.tail(3);
  computeFitts();
  computeMinJerkState();
  computeF();
  err_lyap_ = err_mj_ - x_.head(9);
  updateB();
  updateG();
  K_ev_.block<3, 1>(0, 0) = K_ * err_lyap_;

  // Compute the matrices to put in the QP
  f_QP_ = err_lyap_.transpose() * W_1_ * B_;
  A_QP_ = err_lyap_.transpose() * P_ * B_;
  b_QP_ = -err_lyap_.transpose() * P_ * B_ * K_ev_;
  lb_QP_ << -200, -200, -200, lambda_L_ * (W_ - L_),
      -lambda_tau_ * tau_ - (1 / T_), -200, -200, -200;
  ub_QP_ << 200, 200, 200, lambda_L_ * (max_L_ - L_),
      lambda_tau_ * (0.85 - tau_) - (1 / T_), 200, 200, 200;

  // Solve QP
  bool success = solver_.solve(H_QP_, f_QP_, Eigen::MatrixXd::Zero(0, 0),
                               Eigen::VectorXd::Zero(0), A_QP_, b_QP_, lb_QP_,
                               ub_QP_, false, 1e-6);

  // Handle QP fails
  if (not success) {
    mc_rtc::log::warning(
        "MinimumJerkTask's QP failed,applying worst case convergence");
    u_ = -K_ev_;
  } else {
    u_ = solver_.result();
  }

  dx_ = f_ + g_ * u_;
  x_ = x_ + dx_ * solver.dt();
  x_.tail<3>().normalize();
  commanded_acc_ = commanded_acc_ - dx_.block<3, 1>(6, 0) * solver.dt();

  auto J = jac_->jacobian(robot.mb(), robot.mbc());
  disturbance_acc_ = J * tvm_robot.alphaDExternal();

  // Set PositionTask's refAccel
  PositionTask::refAccel(commanded_acc_ + disturbance_acc_.tail<3>());

  PositionTask::update(solver);
}

void MinimumJerkTask::addToLogger(mc_rtc::Logger &logger) {
  // ========== State logging ========== //
  logger.addLogEntry("MinimumJerkTask_state_error_pos",
                     [this]() -> Eigen::Vector3d { return x_.head<3>(); });
  logger.addLogEntry(
      "MinimumJerkTask_state_error_vel",
      [this]() -> Eigen::Vector3d { return x_.block<3, 1>(3, 0); });
  logger.addLogEntry(
      "MinimumJerkTask_state_error_acc",
      [this]() -> Eigen::Vector3d { return x_.block<3, 1>(6, 0); });
  logger.addLogEntry("MinimumJerkTask_state_traj_length",
                     [this]() -> double { return x_(9); });
  logger.addLogEntry("MinimumJerkTask_state_traj_phase",
                     [this]() -> double { return x_(10); });
  logger.addLogEntry("MinimumJerkTask_state_traj_direction",
                     [this]() -> Eigen::Vector3d { return x_.tail<3>(); });

  // ========== State derivative logging ========== //
  logger.addLogEntry("MinimumJerkTask_stateD_error_vel",
                     [this]() -> Eigen::Vector3d { return dx_.head<3>(); });
  logger.addLogEntry(
      "MinimumJerkTask_stateD_error_acc",
      [this]() -> Eigen::Vector3d { return dx_.block<3, 1>(3, 0); });
  logger.addLogEntry(
      "MinimumJerkTask_stateD_error_jerk",
      [this]() -> Eigen::Vector3d { return dx_.block<3, 1>(6, 0); });
  logger.addLogEntry("MinimumJerkTask_stateD_traj_length",
                     [this]() -> double { return dx_(9); });
  logger.addLogEntry("MinimumJerkTask_stateD_traj_phase",
                     [this]() -> double { return dx_(10); });
  logger.addLogEntry("MinimumJerkTask_stateD_traj_direction",
                     [this]() -> Eigen::Vector3d { return dx_.tail<3>(); });

  // ========== Minimum Jerk traj related ========== //
  logger.addLogEntry("MinimumJerkTask_minimum_jerk_state_error",
                     [this]() -> Eigen::Vector3d { return err_mj_.head<3>(); });
  logger.addLogEntry(
      "MinimumJerkTask_minimum_jerk_state_vel",
      [this]() -> Eigen::Vector3d { return err_mj_.block<3, 1>(3, 0); });
  logger.addLogEntry("MinimumJerkTask_minimum_jerk_state_acc",
                     [this]() -> Eigen::Vector3d { return err_mj_.tail<3>(); });

  // ========== Other ========== //
  logger.addLogEntry("MinimumJerkTask_target_pose",
                     [this]() { return target_pos_; });
  logger.addLogEntry("MinimumJerkTask_control_signal", [this]() { return u_; });
  logger.addLogEntry("MinimumJerkTask_err", [this]() { return eval(); });
  logger.addLogEntry("MinimumJerkTask_err_norm",
                     [this]() { return eval().norm(); });
  logger.addLogEntry("MinimumJerkTask_commanded_acc",
                     [this]() -> Eigen::Vector3d { return commanded_acc_; });
  logger.addLogEntry(
      "MinimumJerkTask_disturb_acc",
      [this]() -> Eigen::Vector3d { return disturbance_acc_.tail<3>(); });
  logger.addLogEntry("MinimumJerkTask_ref_acc", [this]() -> Eigen::Vector3d {
    return commanded_acc_ + disturbance_acc_.tail<3>();
  });
  PositionTask::addToLogger(logger);
}

void MinimumJerkTask::addToGUI(mc_rtc::gui::StateBuilder &gui) {
  gui.addElement(
      {"Tasks", name_, "Visual"},
      mc_rtc::gui::Arrow(
          "MJ Direction",
          mc_rtc::gui::ArrowConfig(mc_rtc::gui::Color(0.0, 0.0, 1.0)),
          [this]() -> Eigen::Vector3d { return curr_pos_; },
          [this]() -> Eigen::Vector3d { return curr_pos_ + 0.2 * D_; }),
      mc_rtc::gui::Arrow(
          "Acc Direction", [this]() -> Eigen::Vector3d { return curr_pos_; },
          [this]() -> Eigen::Vector3d {
            return curr_pos_ - 0.2 * x_.block<3, 1>(6, 0).normalized();
          }));
  gui.addElement(
      {"Tasks", name_, "Parameters"},
      mc_rtc::gui::Label("Trajectory length", [this]() { return L_; }),
      mc_rtc::gui::Label("Trajectory phase", [this]() { return tau_; }),
      mc_rtc::gui::ArrayInput(
          "State weight",
          {"ex", "ey", "ez", "vx", "vy", "vz", "ax", "ay", "az"},
          [this]() { return W_1(); }, [this](Eigen::VectorXd v) { W_1(v); }),
      mc_rtc::gui::ArrayInput(
          "Input weight", {"Jx", "Jy", "Jz", "L", "tau", "Dx", "Dy", "Dz"},
          [this]() { return W_2(); }, [this](Eigen::VectorXd v) { W_2(v); }));
  PositionTask::addToGUI(gui);
}

} // namespace mc_tasks
