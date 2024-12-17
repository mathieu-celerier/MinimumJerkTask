/*
 * Copyright 2015-2024 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <Eigen/Eigenvalues>

#include <mc_tasks/PositionTask.h>

#include <RBDyn/Jacobian.h>
#include <eigen-qld/QLD.h>
#include <mc_rtc/gui/ArrayInput.h>
#include <mc_rtc/gui/ArrayLabel.h>
#include <mc_rtc/gui/Arrow.h>
#include <mc_rtc/gui/Checkbox.h>
#include <mc_rtc/gui/Label.h>
#include <mc_rtc/gui/NumberInput.h>
#include <mc_rtc/gui/Sphere.h>
#include <mc_rtc/gui/Trajectory.h>
#include <mc_tvm/Robot.h>

namespace mc_tasks {

/*! \brief Control the position of a frame */
struct MC_TASKS_DLLAPI MinimumJerkTask : public PositionTask {
public:
  /*! \brief Constructor
   *
   * \param frame Control frame
   *
   * \param stiffness Task stiffness
   *
   * \param weight Task weight
   */
  MinimumJerkTask(const mc_rbdyn::RobotFrame &frame, double weight,
                  bool useFilter = false);

  /*! \brief Constructor
   *
   * \param bodyName Name of the body to control
   *
   * \param robots Robots controlled by this task
   *
   * \param robotIndex Index of the robot controlled by this task
   *
   * \param stiffness Task stiffness
   *
   * \param weight Task weight
   *
   */
  MinimumJerkTask(const std::string &bodyName, const mc_rbdyn::Robots &robots,
                  unsigned int robotIndex, double weight);

  inline void W(double w) {
    if (w <= 0.0)
      mc_rtc::log::error_and_throw<std::domain_error>(
          "W as to be greater than 0");
    W_ = w;
    if (W_ < L_) {
      L_ = W_;
      x_(9) = W_;
    }
  }

  inline double W(void) { return W_; }

  inline void Lu(double len) {
    if (len <= 0.0)
      mc_rtc::log::error_and_throw<std::domain_error>(
          "Length upper bound as to be greater than 0");
    max_L_ = len;
  }

  inline double Lu(void) { return max_L_; }

  inline void lambda_L(double lambda) { lambda_L_ = lambda; }

  inline double lambda_L(void) { return lambda_L_; }

  inline void lambda_tau(double lambda) { lambda_tau_ = lambda; }

  inline double lambda_tau(void) { return lambda_tau_; }

  inline void LQR_Q(Eigen::Vector3d v) {
    LQR_Q_.block<3, 3>(0, 0).diagonal().setConstant(v(0));
    LQR_Q_.block<3, 3>(3, 3).diagonal().setConstant(v(1));
    LQR_Q_.block<3, 3>(6, 6).diagonal().setConstant(v(2));
    solveLQR();
    computeQ();
  }

  inline Eigen::Vector3d LQR_Q(void) {
    return Eigen::Vector3d(LQR_Q_(0, 0), LQR_Q_(3, 3), LQR_Q_(6, 6));
  }

  inline void LQR_R(double w) {
    LQR_R_.diagonal().setConstant(w);
    solveLQR();
    computeQ();
  }

  inline double LQR_R(void) { return LQR_R_(0, 0); }

  inline void W_e(Eigen::Vector3d W) {
    W_e_.block<3, 3>(0, 0).diagonal().setConstant(W(0));
    W_e_.block<3, 3>(3, 3).diagonal().setConstant(W(1));
    W_e_.block<3, 3>(6, 6).diagonal().setConstant(W(2));
  }

  inline Eigen::Vector3d W_e(void) {
    return Eigen::Vector3d(W_e_(0, 0), W_e_(3, 3), W_e_(6, 6));
  }

  inline void W_u(Eigen::Vector4d W) {
    W_u_.block<3, 3>(0, 0).diagonal().setConstant(W(0));
    W_u_(3, 3) = W(1);
    W_u_(4, 4) = W(2);
    W_u_.block<3, 3>(5, 5).diagonal().setConstant(W(3));
    H_QP_ = 2 * W_u_.transpose() * W_u_;
  }

  inline Eigen::Vector4d W_u(void) {
    return Eigen::Vector4d(W_u_(0, 0), W_u_(3, 3), W_u_(4, 4), W_u_(5, 5));
  }

  inline void K(Eigen::Matrix<double, 3, 9> K) {
    K_ = K;
    computeQ();
  }

  inline Eigen::Matrix<double, 3, 9> K(void) { return K_; }

  inline void P(Eigen::MatrixXd P) { P_ = P; }

  inline Eigen::MatrixXd P(void) { return P_; }

  inline void Q(Eigen::MatrixXd Q) { Q_ = Q; }

  inline Eigen::MatrixXd Q(void) { return Q_; }

  inline void computeQ(void) { Q_ = K_.transpose() * LQR_R_ * K_ + LQR_Q_; }

  inline void set_vel_filter_tau(double tau) { vel_filtering_tau_ = tau; }

  inline void fitts_a(double a) { fitts_a_ = a; }

  inline double fitts_a(void) { return fitts_a_; }

  inline void fitts_b(double b) { fitts_b_ = b; }

  inline double fitts_b(void) { return fitts_b_; }

  inline void react_time(double time) { reaction_time_ = time; }

  inline double react_time(void) { return reaction_time_; }

  inline void set_gamma(double g) { gamma_state_ = g; };

  inline void setTarget(Eigen::Vector3d pos, bool reset = true) {
    target_pos_ = pos;
    init_ = reset;
    PositionTask::position(pos);
  }

  inline Eigen::Vector3d getTarget(void) { return target_pos_; }

  void addToGUI(mc_rtc::gui::StateBuilder &gui) override;

  inline Eigen::Vector3d eval(void) { return x_.head<3>(); };
  inline Eigen::Vector3d speed(void) { return x_.block<3, 1>(3, 0); };

protected:
  mc_rbdyn::ConstRobotFramePtr frame_;
  void addToLogger(mc_rtc::Logger &logger) override;

  void update(mc_solver::QPSolver &solver) override;

  void computeDuration(void);
  void computeMinJerkState(void);
  void computeF(void);
  void updateB(void);
  void updateG(void);

  void solveLQR(void);

  std::vector<Eigen::Vector3d> computeVelTraj(void);

  std::string bodyName_;
  Eigen::Vector3d target_pos_;
  Eigen::Vector3d curr_pos_;

  bool init_;
  double gamma_state_;
  double gamma_output_;
  std::string qp_state;

  // Control parameters
  double W_;
  double max_L_;
  double max_tau_;
  double max_jerk_;
  double max_omega_;
  double lambda_L_;
  double lambda_tau_;
  Eigen::Matrix<double, 9, 9> LQR_Q_;
  Eigen::Matrix<double, 3, 3> LQR_R_;
  Eigen::Matrix<double, 9, 9> W_e_;
  Eigen::Matrix<double, 8, 8> W_u_;
  Eigen::Matrix<double, 3, 9> K_;
  Eigen::Matrix<double, 9, 9> P_;
  Eigen::Matrix<double, 9, 9> Q_;
  double fitts_a_;
  double fitts_b_;
  double reaction_time_;
  double max_jac_tau_;
  double lambda_jac_L_;
  double lambda_jac_tau_;
  double lambda_jac_D_;
  double gain_linear_cost_;
  double vel_filtering_tau_;
  bool filter_velocity_;

  // Control variables
  double L_;
  double tau_;
  Eigen::Matrix<double, 14, 1> x_;
  Eigen::Matrix<double, 14, 1> dx_;
  Eigen::Matrix<double, 14, 1> f_;
  Eigen::Matrix<double, 14, 8> g_;
  Eigen::Matrix<double, 8, 1> u_;
  Eigen::Matrix<double, 9, 9> A_;
  Eigen::Matrix<double, 9, 8> B_;
  double T_fitts_;
  double T_;
  Eigen::Matrix<double, 9, 1> err_mj_;
  Eigen::Matrix<double, 9, 1> err_lyap_;
  Eigen::Matrix<double, 8, 1> K_ev_;
  Eigen::Vector3d D_;
  Eigen::Vector3d commanded_acc_;
  Eigen::Vector3d ref_acc_;
  Eigen::Vector6d disturbance_acc_;
  Eigen::Vector3d mj_pose_;
  double reaction_time_counter_;
  Eigen::Matrix<double, 9, 1> dev_pred_;
  Eigen::Matrix<double, 9, 1> dev_diff_;
  Eigen::Matrix<double, 9, 1> dyn_error_;
  Eigen::Vector3d filtered_vel_;
  Eigen::Vector3d filtered_acc_;
  Eigen::Vector3d vel_hat;

  // QP matrices
  Eigen::Matrix<double, 8, 8> H_QP_;
  Eigen::Matrix<double, 8, 1> f_QP_;
  Eigen::Matrix<double, 1, 8> A_QP_;
  Eigen::Matrix<double, 1, 1> b_QP_;
  Eigen::Matrix<double, 8, 1> lb_QP_;
  Eigen::Matrix<double, 8, 1> ub_QP_;

  bool filter_out;

  // External forces
  rbd::Jacobian *jac_;

  // QP solver
  Eigen::QLD solver_;
};

} // namespace mc_tasks
