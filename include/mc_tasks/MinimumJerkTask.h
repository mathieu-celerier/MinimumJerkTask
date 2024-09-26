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
  MinimumJerkTask(const mc_rbdyn::RobotFrame &frame, double weight);

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

  inline void W_1(Eigen::VectorXd W) {
    W_1_ = W.asDiagonal();
    solveLQR();
    computeQ();
  }

  inline Eigen::VectorXd W_1(void) { return W_1_.diagonal(); }

  inline void W_2(Eigen::VectorXd W) {
    W_2_ = W.asDiagonal();
    H_QP_ = 2 * W_2_.transpose() * W_2_;
    solveLQR();
    computeQ();
  }

  inline Eigen::VectorXd W_2(void) { return W_2_.diagonal(); }

  inline void K(Eigen::Matrix<double, 3, 9> K) {
    K_ = K;
    computeQ();
  }

  inline Eigen::Matrix<double, 3, 9> K(void) { return K_; }

  inline void P(Eigen::MatrixXd P) { P_ = P; }

  inline Eigen::MatrixXd P(void) { return P_; }

  inline void Q(Eigen::MatrixXd Q) { Q_ = Q; }

  inline Eigen::MatrixXd Q(void) { return Q_; }

  inline void computeQ(void) {
    Q_ = K_.transpose() * W_2_.block<3, 3>(0, 0) * K_ + W_1_;
  }

  inline void fitts_a(double a) { fitts_a_ = a; }

  inline double fitts_a(void) { return fitts_a_; }

  inline void fitts_b(double b) { fitts_b_ = b; }

  inline double fitts_b(void) { return fitts_b_; }

  inline void react_time(double time) { reaction_time_ = time; }

  inline double react_time(void) { return reaction_time_; }

  inline void setTarget(Eigen::Vector3d pos) {
    target_pos_ = pos;
    init_ = true;
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
  void computeLQRGain(void);

  std::string bodyName_;
  Eigen::Vector3d target_pos_;
  Eigen::Vector3d curr_pos_;

  bool init_;
  bool dist_acc_before_;
  double compensation_factor_;
  std::string qp_state;

  // Control parameters
  double W_;
  double max_L_;
  double max_tau_;
  double lambda_L_;
  double lambda_tau_;
  Eigen::Matrix<double, 9, 9> W_1_;
  Eigen::Matrix<double, 8, 8> W_2_;
  Eigen::Matrix<double, 3, 9> K_;
  Eigen::Matrix<double, 9, 9> P_;
  Eigen::Matrix<double, 9, 9> Q_;
  double fitts_a_;
  double fitts_b_;
  double reaction_time_;

  sva::MotionVecd acc_body_;
  sva::MotionVecd vel_body_;

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

  // QP matrices
  Eigen::Matrix<double, 8, 8> H_QP_;
  Eigen::Matrix<double, 8, 1> f_QP_;
  Eigen::Matrix<double, 1, 8> A_QP_;
  Eigen::Matrix<double, 1, 1> b_QP_;
  Eigen::Matrix<double, 8, 1> lb_QP_;
  Eigen::Matrix<double, 8, 1> ub_QP_;

  // External forces
  rbd::Jacobian *jac_;

  // QP solver
  Eigen::QLD solver_;
};

} // namespace mc_tasks
