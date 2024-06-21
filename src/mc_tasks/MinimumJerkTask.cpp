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
      bodyName_(frame_->body()), W_(0.03), max_L_(2.0), lambda_L_(100),
      lambda_tau_(100), fitts_a_(0), fitts_b_(1.0), solver_() {
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
  g_.block<2, 2>(9, 0).setIdentity();

  L_ = 0.5;
  x_(10) = L_;
  tau_ = 0.0;
  x_(11) = tau_;
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
      (L_ / pow(T_, 2)) * (-120.0 * tau3 + 180.0 * tau2 - 60.0) * D_;
}

void MinimumJerkTask::computeF(void) {
  f_.block<3, 1>(0, 0) = x_.block<3, 1>(3, 0);
  f_.block<3, 1>(3, 0) = x_.block<3, 1>(6, 0);
  f_.block<3, 1>(6, 0) =
      (L_ / pow(T_, 3)) * (-360.0 * tau_ * tau_ + 360.0 * tau_ - 60.0) * D_;
  f_(11) = 1 / T_;
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

void MinimumJerkTask::update(mc_solver::QPSolver &solver) {
  auto &robot = frame_->robot();
  sva::PTransform transform(robot.bodyPosW(bodyName_));
  Eigen::Vector3d pos = robot.bodyPosW(bodyName_).translation();
  Eigen::Vector3d vel = robot.bodyVelW(bodyName_).linear();
  Eigen::Vector3d acc =
      transform.rotation().transpose() * robot.bodyAccB(bodyName_).linear() +
      robot.bodyVelW(bodyName_).angular().cross(vel);

  Eigen::Vector3d err = target_pos_ - pos;
  x_.block<3, 1>(0, 0) = err;
  x_.block<3, 1>(3, 0) = -vel;
  x_.block<3, 1>(6, 0) = -acc;

  L_ = x_(10);
  tau_ = x_(11);
  computeFitts();
  computeMinJerkState();
  err_lyap_ = err_mj_ - x_.head(9);
  updateB();
  updateG();
  K_ev_.block<3, 1>(0, 0) = K_ * err_lyap_;

  // Compute the matrices to put in the QP
  f_QP_ = err_lyap_.transpose() * W_1_ * B_;
  A_QP_ = err_lyap_.transpose() * P_ * B_;
  b_QP_ = -err_lyap_.transpose() * P_ * B_ * K_ev_;
  lb_QP_ << -1000, -1000, -1000, -lambda_L_ * (W_ - L_),
      -lambda_tau_ * tau_ - (1 / T_), -1000, -1000, -1000;
  ub_QP_ << 1000, 1000, 1000, lambda_L_ * (max_L_ - L_),
      lambda_tau_ * (1 - tau_) - (1 / T_), 1000, 1000, 1000;

  // Solve QP
  solver_.solve(H_QP_, f_QP_, Eigen::MatrixXd::Zero(0, 0),
                Eigen::VectorXd::Zero(0), A_QP_, b_QP_, lb_QP_, ub_QP_, false,
                1e-6);

  // Handle QP fails
  // if (qp.failed()) mc_rtc::log::warning("MinimumJerkTask's QP failed,
  // applying worst case convergence");

  dx_ = f_ + g_ * u_;
  x_ = x_ + dx_ * solver.dt();

  // Set PositionTask's refAccel
  PositionTask::refAccel(x_.block<3, 1>(6, 0));

  PositionTask::update(solver);
}

void MinimumJerkTask::addToLogger(mc_rtc::Logger &logger) {
  PositionTask::addToLogger(logger);
}

void MinimumJerkTask::addToGUI(mc_rtc::gui::StateBuilder &gui) {
  PositionTask::addToGUI(gui);
}

} // namespace mc_tasks
