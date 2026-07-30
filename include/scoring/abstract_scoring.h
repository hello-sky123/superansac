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

#include <Eigen/Core>

#include <cmath>
#include <utility>
#include <vector>

#include "../estimators/abstract_estimator.h"
#include "../models/model.h"
#include "../utils/macros.h"
#include "../utils/types.h"
#include "score.h"

namespace superansac::scoring {

class AbstractScoring {
 public:
  // Constructor
  AbstractScoring() : threshold(1.0), squaredThreshold(1.0) {}

  // Destructor
  virtual ~AbstractScoring() = default;

  // Set the threshold
  FORCE_INLINE virtual void setThreshold(double kThreshold_) = 0;

  // Get the threshold
  [[nodiscard]] FORCE_INLINE const double& getThreshold() const { return threshold; };

  FORCE_INLINE virtual void updateSPRTParameters(const Score& currentBest, int iterationIndex,
                                                 size_t totalPoints) = 0;

  // Sample function.
  // Non-virtual: the default arguments live here, in one place, and the
  // overridable part is scoreImpl() below. Defaults on a virtual are bound
  // statically from the declared type, so a subclass that repeated them with
  // different values would silently change behaviour depending on whether the
  // call went through the base or the derived type.
  FORCE_INLINE Score score(const DataMatrix& kData_,                 // Data matrix
                           const models::Model& kModel_,             // The model to be scored
                           const estimator::Estimator* kEstimator_,  // Estimator
                           std::vector<size_t>& inliers_,            // Inlier indices
                           const bool kStoreInliers_ = true,         // Store inliers or not
                           const Score& kBestScore_ = Score(),       // The best score found so far
                           std::vector<const std::vector<size_t>*>* kPotentialInlierSets_ =
                               nullptr) const  // The potential inlier sets from the inlier selector
  {
    return scoreImpl(kData_, kModel_, kEstimator_, inliers_, kStoreInliers_, kBestScore_,
                     kPotentialInlierSets_);
  }

  // Get weights for the points. Non-virtual wrapper, see score() above.
  FORCE_INLINE void getWeights(
      const DataMatrix& kData_,                              // Data matrix
      const models::Model& kModel_,                          // The model to be scored
      const estimator::Estimator* kEstimator_,               // Estimator
      std::vector<double>& weights_,                         // The weights of the points
      const std::vector<size_t>* kIndices_ = nullptr) const  // The indices of the points
  {
    getWeightsImpl(kData_, kModel_, kEstimator_, weights_, kIndices_);
  }

  static FORCE_INLINE void getInliers(
      const DataMatrix& kData_,                          // Data matrix
      const models::Model& kModel_,                      // The model to be scored
      const estimator::Estimator* kEstimator_,           // Estimator
      std::vector<std::pair<double, size_t>>& inliers_,  // The inliers of the model
      const double kThreshold_,                          // The threshold for inlier selection
      const bool kReturnSquaredResidual = true)          // Return the squared residuals or not
  {
    // The number of points
    const long kPointNumber = kData_.rows();

    // Select against the caller-supplied threshold, not the member one, so a
    // caller can widen or narrow the selection (IteratedLMEDSOptimizer passes
    // thresholdMultiplier * getThreshold() to collect a wider candidate set).
    // Residuals below are squared, so square the threshold rather than taking a
    // root per point.
    const double kSquaredThreshold = kThreshold_ * kThreshold_;

    // Iterate through all points, calculate the squaredResiduals and store the points as inliers if needed.
    inliers_.clear();
    inliers_.reserve(kPointNumber);
    for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx) {
      // The squared residual, Calculate the point-to-model residual
      double squaredResidual = kEstimator_->squaredResidual(kData_.row(pointIdx).data(), kModel_);

      // If the residual is smaller than the threshold, store it as an inlier and
      // increase the score.
      if (squaredResidual < kSquaredThreshold) {
        if (kReturnSquaredResidual)
          inliers_.emplace_back(squaredResidual, pointIdx);
        else
          inliers_.emplace_back(std::sqrt(squaredResidual), pointIdx);
      }
    }
  }

 protected:
  // Overridable implementations. No default arguments here: every parameter is
  // supplied by the non-virtual wrappers above, so the defaults cannot drift
  // between base and derived declarations.
  FORCE_INLINE virtual Score scoreImpl(
      const DataMatrix& kData_,                 // Data matrix
      const models::Model& kModel_,             // The model to be scored
      const estimator::Estimator* kEstimator_,  // Estimator
      std::vector<size_t>& inliers_,            // Inlier indices
      bool kStoreInliers_,                      // Store inliers or not
      const Score& kBestScore_,                 // The best score found so far
      std::vector<const std::vector<size_t>*>* kPotentialInlierSets_)
      const = 0;  // The potential inlier sets from the inlier selector

  FORCE_INLINE virtual void getWeightsImpl(
      const DataMatrix& kData_,                         // Data matrix
      const models::Model& kModel_,                     // The model to be scored
      const estimator::Estimator* kEstimator_,          // Estimator
      std::vector<double>& weights_,                    // The weights of the points
      const std::vector<size_t>* kIndices_) const = 0;  // The indices of the points

  // 阈值
  double threshold;
  double squaredThreshold;
};

}  // namespace superansac::scoring
