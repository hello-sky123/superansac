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
//     * Neither the name of ETH Zurich nor the
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
// Author: Daniel Barath (majti89@gmail.com)
#pragma once

#include <Eigen/Eigen>

#include "abstract_solver.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac {
namespace estimator {
namespace solver {
// Plane-and-parallax fundamental matrix solver (used by DEGENSAC).
// Given a homography H induced by the dominant plane, two off-plane
// correspondences determine the epipole as the intersection of the
// lines connecting the H-projected and observed points; then
// F = [e']_x * H. Ported from graph-cut-ransac's
// FundamentalMatrixPlaneParallaxSolver.
class FundamentalMatrixPlaneParallaxSolver : public AbstractSolver {
 protected:
  const ModelMatrix* homography;  // Not owned; set per DEGENSAC event

 public:
  FundamentalMatrixPlaneParallaxSolver() : homography(nullptr) {}

  ~FundamentalMatrixPlaneParallaxSolver() {}

  void setHomography(const ModelMatrix* kHomography_) { homography = kHomography_; }

  // The maximum number of solutions returned by the estimator
  size_t maximumSolutions() const override { return 1; }

  // The minimum number of points required for the estimation
  size_t sampleSize() const override { return 2; }

  FORCE_INLINE bool estimateModel(
      const DataMatrix& kData_,                          // The set of data points
      const size_t* kSample_,                            // The sample used for the estimation
      const size_t kSampleNumber_,                       // The size of the sample
      std::vector<models::Model>& models_,               // The estimated model parameters
      const double* kWeights_ = nullptr) const override  // The weight for each point
  {
    // Check if the required homography has been set
    if (homography == nullptr) return false;

    // The rows of the two sampled correspondences
    const double* kPoint1 = kData_.data() + kSample_[0] * kData_.cols();
    const double* kPoint2 = kData_.data() + kSample_[1] * kData_.cols();

    const Eigen::Vector3d sourcePoint1(kPoint1[0], kPoint1[1], 1),
        destinationPoint1(kPoint1[2], kPoint1[3], 1), sourcePoint2(kPoint2[0], kPoint2[1], 1),
        destinationPoint2(kPoint2[2], kPoint2[3], 1);

    // The homography as a fixed-size matrix for the products below
    const Eigen::Matrix3d kHomography = homography->block<3, 3>(0, 0);

    // Projecting the points by the homography matrix
    const Eigen::Vector3d projectedPoint1 = kHomography * sourcePoint1,
                          projectedPoint2 = kHomography * sourcePoint2;

    // Calculating the parameters of the lines between the projected and original points
    const Eigen::Vector3d line1 = projectedPoint1.cross(destinationPoint1),
                          line2 = projectedPoint2.cross(destinationPoint2);

    // Estimating the epipole
    const Eigen::Vector3d epipole = line1.cross(line2);

    // There is no intersection (e.g., both points lie on the plane)
    if (std::abs(epipole(2)) < std::numeric_limits<double>::epsilon()) return false;

    // Calculate the cross-product matrix of the epipole
    Eigen::Matrix3d epipolarCross;
    epipolarCross << 0, -epipole(2), epipole(1), epipole(2), 0, -epipole(0), -epipole(1),
        epipole(0), 0;

    // Calculate the fundamental matrix
    const Eigen::Matrix3d kFundamentalMatrix = epipolarCross * kHomography;
    if (!kFundamentalMatrix.allFinite()) return false;

    models::Model model;
    model.getMutableData() = kFundamentalMatrix;
    models_.emplace_back(model);
    return true;
  }
};
}  // namespace solver
}  // namespace estimator
}  // namespace superansac
