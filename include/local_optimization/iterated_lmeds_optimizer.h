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

#include <Eigen/Core>

#include <vector>

#include "../models/model.h"
#include "../utils/types.h"
#include "abstract_local_optimizer.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core.hpp"

namespace superansac {
namespace local_optimization {
// Templated class for estimating a model for RANSAC. This class is purely a
// virtual class and should be implemented for the specific task that RANSAC is
// being used for. Two methods must be implemented: estimateModel and residual. All
// other methods are optional, but will likely enhance the quality of the RANSAC
// output.
class IteratedLMEDSOptimizer : public LocalOptimizer {
 protected:
  size_t binNumber;
  models::Types modelType;

 public:
  IteratedLMEDSOptimizer() : binNumber(50), modelType(models::Types::Homography) {}

  ~IteratedLMEDSOptimizer() {}

  // Set the maximum number of iterations
  void setMaxIterations(const size_t maxIterations_) {}

  void setModelType(const models::Types kModelType_) { modelType = kModelType_; }

  // The function for estimating the model parameters from the data points.
  void run(const DataMatrix& kData_,              // The data points
           const std::vector<size_t>& kInliers_,  // The inliers of the previously estimated model
           const models::Model& kModel_,          // The previously estimated model
           const scoring::Score& kScore_,         // The of the previously estimated model
           const estimator::Estimator* kEstimator_,  // The estimator used for the model estimation
           scoring::AbstractScoring* kScoring_,  // The scoring object used for the model estimation
           models::Model& estimatedModel_,       // The estimated model
           scoring::Score& estimatedScore_,      // The score of the estimated model
           std::vector<size_t>& estimatedInliers_) const  // The inliers of the estimated model
  {
    // The invalid score
    static const scoring::Score kInvalidScore = scoring::Score();

    // The threshold for the candidate inlier selection below
    const double kSelectionThreshold = kScoring_->getThreshold();

    // The estimated models
    std::vector<models::Model> estimatedModels;
    scoring::Score currentScore = kInvalidScore;
    std::vector<double> weights(kData_.rows());

    // Initialize the estimated model and score
    estimatedModel_ = kModel_;
    estimatedScore_ = kScore_;

    // Clear the estimated inliers
    estimatedInliers_.clear();
    estimatedInliers_.reserve(kData_.rows());

    // Temp inliers for selecting the best model
    std::vector<size_t> tmpInliers;
    tmpInliers.reserve(kData_.rows());

    // A flag indicating if the model has been updated
    bool updated = false;

    std::vector<std::pair<double, size_t>> candidateInliers;
    candidateInliers.reserve(kData_.rows());

    // Collect the candidate inliers, sorted by residual, to be split into bins
    kScoring_->getInliers(kData_, estimatedModel_, kEstimator_, candidateInliers,
                          kSelectionThreshold);

    // Sort the inliers by the residuals
    std::sort(std::begin(candidateInliers), std::end(candidateInliers));

    // Divide the inliers into bins
    const size_t binSize = candidateInliers.size() / binNumber;

    std::vector<cv::Point2d> sourceKeypoints, destinationKeypoints;
    std::vector<uchar> mask;
    mask.reserve(candidateInliers.size());
    sourceKeypoints.reserve(candidateInliers.size());
    destinationKeypoints.reserve(candidateInliers.size());

    for (size_t binIdx = 0; binIdx < binNumber; ++binIdx) {
      // The end indices of the current bin
      const size_t startIdx = binIdx * binSize;
      const size_t endIdx = std::min((binIdx + 1) * binSize, candidateInliers.size());

      // If the end index is smaller than the sample size, continue
      if (endIdx < kEstimator_->sampleSize() + 1) continue;

      // Add the new inliers to the keypoints
      sourceKeypoints.resize(endIdx);
      destinationKeypoints.resize(endIdx);
      for (size_t inlierIdx = startIdx; inlierIdx < endIdx; ++inlierIdx) {
        const auto& eigenCorrespondence = kData_.row(candidateInliers[inlierIdx].second);

        sourceKeypoints[inlierIdx].x = eigenCorrespondence(0);
        sourceKeypoints[inlierIdx].y = eigenCorrespondence(1);
        destinationKeypoints[inlierIdx].x = eigenCorrespondence(2);
        destinationKeypoints[inlierIdx].y = eigenCorrespondence(3);
      }
      mask.resize(endIdx);

      cv::Mat cvModel;
      if (modelType == models::Types::Homography)
        cvModel = cv::findHomography(sourceKeypoints, destinationKeypoints, cv::LMEDS, 3.0, mask);
      else if (modelType == models::Types::FundamentalMatrix)
        cvModel = cv::findFundamentalMat(sourceKeypoints, destinationKeypoints, cv::LMEDS);
      else
        throw std::runtime_error("The model type is not supported.");

      if (cvModel.empty()) continue;

      // Convert the model to an Eigen matrix
      models::Model model;
      auto& data = model.getMutableData();
      data.resize(3, 3);
      data << cvModel.at<double>(0, 0), cvModel.at<double>(0, 1), cvModel.at<double>(0, 2),
          cvModel.at<double>(1, 0), cvModel.at<double>(1, 1), cvModel.at<double>(1, 2),
          cvModel.at<double>(2, 0), cvModel.at<double>(2, 1), cvModel.at<double>(2, 2);

      // Calculate the score of the estimated model
      tmpInliers.clear();
      currentScore = kScoring_->score(kData_, model, kEstimator_, tmpInliers);

      // Check if the current model is better than the previous one
      if (currentScore > estimatedScore_) {
        // Update the estimated model
        estimatedModel_ = model;
        estimatedScore_ = currentScore;
        tmpInliers.swap(estimatedInliers_);
      }
    }
  }
};
}  // namespace local_optimization
}  // namespace superansac