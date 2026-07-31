// Copyright (C) 2024 ETH Zurich.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include "numerical_optimizer/camera_pose.h"

#include "../models/model.h"
#include "../utils/math_utils.h"
#include "../utils/types.h"
#include "abstract_solver.h"
#include "numerical_optimizer/bundle.h"
#include "numerical_optimizer/essential.h"
#include "solver_essential_matrix_five_point_nister.h"
#include "solver_fundamental_matrix_eight_point.h"

namespace superansac {
namespace estimator {
namespace solver {
// Non-minimal solver refining an essential matrix over six or more
// correspondences by nonlinear least squares. Returns a single solution.
class EssentialMatrixBundleAdjustmentSolver : public AbstractSolver {
 protected:
  poselib::BundleOptions options;
  size_t pointNumberForCheiralityCheck;
  const std::vector<double>* pointWeights;

 public:
  EssentialMatrixBundleAdjustmentSolver(poselib::BundleOptions kOptions_ = poselib::BundleOptions())
      : options(kOptions_), pointNumberForCheiralityCheck(1), pointWeights(nullptr) {}

  ~EssentialMatrixBundleAdjustmentSolver() {}

  // Determines if there is a chance of returning multiple models
  // the function 'estimateModel' is applied.
  bool returnMultipleModels() const override { return maximumSolutions() > 1; }

  // The maximum number of solutions returned by the estimator
  size_t maximumSolutions() const override { return 1; }

  // The minimum number of points required for the estimation
  size_t sampleSize() const override { return 6; }

  poselib::BundleOptions& getMutableOptions() { return options; }

  void setWeights(const std::vector<double>* pointWeights_) { pointWeights = pointWeights_; }

  void setCheiralityPointNumber(const size_t kPointNumber_) {
    pointNumberForCheiralityCheck = kPointNumber_;
  }

  // Estimate the model parameters from the given point sample
  // using weighted fitting if possible.
  FORCE_INLINE bool estimateModelImpl(
      const DataMatrix& kData_,                 // The set of data points
      const size_t* kSample_,                   // The sample used for the estimation
      const size_t kSampleNumber_,              // The size of the sample
      std::vector<models::Model>& models_,      // The estimated model parameters
      const double* kWeights_) const override;  // The weight for each point
};

FORCE_INLINE bool EssentialMatrixBundleAdjustmentSolver::estimateModelImpl(
    const DataMatrix& kData_,             // The set of data points
    const size_t* kSample_,               // The sample used for the estimation
    const size_t kSampleNumber_,          // The size of the sample
    std::vector<models::Model>& models_,  // The estimated model parameters
    const double* kWeights_) const        // The weight for each point
{
  // Check if we have enough points for the bundle adjustment
  if (kSampleNumber_ < sampleSize()) return false;

  // The point correspondences. Thread-local scratch buffers: this solver
  // runs in the inner loops of the local optimizers, so reusing the
  // buffers avoids per-call heap allocations. The contents are fully
  // overwritten below.
  static thread_local std::vector<Eigen::Vector2d> x1;
  static thread_local std::vector<Eigen::Vector2d> x2;
  static thread_local std::vector<double> weights;
  x1.resize(kSampleNumber_);
  x2.resize(kSampleNumber_);
  weights.clear();
  // Per-constraint confidences fed into the relative-pose LM bundle
  // adjustment. The kWeights_ array passed by the local optimizers
  // (e.g. the MAGSAC marginalized weights aligned with the sample
  // order) takes precedence; the member pointWeights is the legacy
  // fallback. Previously kWeights_ was silently ignored, so the LM
  // always ran with uniform per-point weights.
  const bool useArgWeights = kWeights_ != nullptr;
  const bool useMemberWeights = !useArgWeights && pointWeights != nullptr;
  if (useArgWeights || useMemberWeights) weights.resize(kSampleNumber_);

  // Filling the point correspondences if the sample is not provided
  if (kSample_ == nullptr) {
    // Filling the point correspondences
    for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++) {
      x1[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 0), kData_(pointIdx, 1));
      x2[pointIdx] = Eigen::Vector2d(kData_(pointIdx, 2), kData_(pointIdx, 3));
      if (useArgWeights)
        weights[pointIdx] = kWeights_[pointIdx];
      else if (useMemberWeights)
        weights[pointIdx] = (*pointWeights)[pointIdx];
    }
  } else  // Filling the point correspondences if the sample is provided
  {
    for (size_t pointIdx = 0; pointIdx < kSampleNumber_; pointIdx++) {
      const size_t& idx = kSample_[pointIdx];
      x1[pointIdx] = Eigen::Vector2d(kData_(idx, 0), kData_(idx, 1));
      x2[pointIdx] = Eigen::Vector2d(kData_(idx, 2), kData_(idx, 3));
      if (useArgWeights)
        weights[pointIdx] = kWeights_[pointIdx];
      else if (useMemberWeights)
        weights[pointIdx] = (*pointWeights)[idx];
    }
  }

  // Estimating the essential matrix using the five-point algorithm if no model is provided
  if (models_.size() == 0) {
    // Initializing the five-point solver
    EssentialMatrixFivePointNisterSolver fivePointSolver;
    // Estimating the essential matrix
    fivePointSolver.estimateModel(kData_, kSample_, kSampleNumber_, models_);

    // If the estimation failed, return false
    if (models_.size() == 0) return false;
  }

  // The options for the bundle adjustment
  poselib::BundleOptions tmpOptions = options;
  // If the sample is provided, we use a more robust loss function. This typically runs in the end of the robust estimation
  if (kSample_ != nullptr) {
    tmpOptions.loss_scale = 0.5 * options.loss_scale;
    tmpOptions.max_iterations = 100;
    tmpOptions.loss_type = poselib::BundleOptions::LossType::CAUCHY;
  }

  // Build the (normalized) bearing vectors used to disambiguate the
  // four essential-matrix decompositions by cheirality. The sample is
  // score-sorted (PROSAC), so the first points are the highest
  // confidence. Using several points and a robust majority VOTE --
  // instead of the previous single-point all-or-nothing filter --
  // reliably selects the correct pose without being defeated by one
  // bad correspondence (the old all-AND filter degraded as the point
  // count grew, since a single outlier rejected the true pose).
  const size_t kPointNumberForCheck =
      std::max<size_t>(1, std::min(pointNumberForCheiralityCheck, kSampleNumber_));
  static thread_local std::vector<Eigen::Vector3d> x1CheiralityCheck, x2CheiralityCheck;
  x1CheiralityCheck.resize(kPointNumberForCheck);
  x2CheiralityCheck.resize(kPointNumberForCheck);
  for (size_t idx = 0; idx < kPointNumberForCheck; idx++) {
    const size_t& pointIdx = kSample_ == nullptr ? idx : kSample_[idx];
    x1CheiralityCheck[idx] = Eigen::Vector3d(kData_(pointIdx, 0), kData_(pointIdx, 1), 1);
    x2CheiralityCheck[idx] = Eigen::Vector3d(kData_(pointIdx, 2), kData_(pointIdx, 3), 1);
    x1CheiralityCheck[idx].normalize();
    x2CheiralityCheck[idx].normalize();
  }

  // Empty filter so motion_from_essential returns all four candidate poses.
  static const std::vector<Eigen::Vector3d> kEmptyCheck;

  // The selected pose: prefer more cheirality votes, then lower BA cost.
  size_t bestVotes = 0;
  double bestCost = std::numeric_limits<double>::max();
  poselib::CameraPose bestPose;
  bool haveBestPose = false;

  // Iterating through the potential models.
  for (auto& model : models_) {
    // Decompose the essential matrix to all four candidate poses
    poselib::CameraPoseVector poses;
    poselib::motion_from_essential(model.getData().block<3, 3>(0, 0),  // The essential matrix
                                   kEmptyCheck, kEmptyCheck,  // No internal filtering; vote below
                                   &poses);                   // The decomposed poses

    // Robust cheirality vote: count the points in front of both
    // cameras for each candidate pose.
    size_t maxVotes = 0;
    for (const auto& pose : poses) {
      size_t votes = 0;
      for (size_t i = 0; i < kPointNumberForCheck; ++i)
        if (poselib::check_cheirality(pose, x1CheiralityCheck[i], x2CheiralityCheck[i])) ++votes;
      if (votes > maxVotes) maxVotes = votes;
    }

    // Refine the pose(s) with the most cheirality votes and keep the
    // best by (votes, then BA cost). Refining only the top-voted
    // pose(s) is typically a single refinement -- fewer than before.
    for (auto& pose : poses) {
      size_t votes = 0;
      for (size_t i = 0; i < kPointNumberForCheck; ++i)
        if (poselib::check_cheirality(pose, x1CheiralityCheck[i], x2CheiralityCheck[i])) ++votes;
      if (votes < maxVotes) continue;

      // Perform the bundle adjustment.
      // NOTE: the returned BundleStats must be captured; previously the
      // return value was discarded and the uninitialized local `stats.cost`
      // was read, making the best-pose selection undefined behavior.
      poselib::BundleStats stats = poselib::refine_relpose(x1, x2, &pose, tmpOptions, weights);

      if (!haveBestPose || votes > bestVotes || (votes == bestVotes && stats.cost < bestCost)) {
        bestVotes = votes;
        bestCost = stats.cost;
        bestPose = pose;
        haveBestPose = true;
      }
    }
  }

  // Composing the essential matrix from the pose
  if (haveBestPose) {
    Eigen::Matrix3d essentialMatrix;
    poselib::essential_from_motion(bestPose, &essentialMatrix);

    // Adding the essential matrix as the estimated models.
    models_.resize(1);
    models_[0].getMutableData() = essentialMatrix;
  }

  return models_.size() > 0;
}
}  // namespace solver
}  // namespace estimator
}  // namespace superansac