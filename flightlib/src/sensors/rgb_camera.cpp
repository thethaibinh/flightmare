#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

RGBCamera::RGBCamera()
  : channels_(3),
    width_(640),
    height_(480),
    fov_{90.0},
    depth_scale_{0.2},
    T_BC_{(Matrix<4, 4>() << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
            .finished()},
    K_{(Matrix<3, 3>() << 174.246, 0.0, 320.0, 0.0, 130.684, 240.0, 0.0, 0.0,
        1.0)
         .finished()},
    enabled_layers_({false, false, false}) {}

RGBCamera::~RGBCamera() {}

bool RGBCamera::feedImageQueue(const int image_layer,
                               const cv::Mat& image_mat) {
  queue_mutex_.lock();
  switch (image_layer) {
    case 0:  // rgb image
      if (rgb_queue_.size() > queue_size_) rgb_queue_.resize(queue_size_);
      rgb_queue_.push_back(image_mat);
      break;
    case CameraLayer::DepthMap:
      if (depth_queue_.size() > queue_size_) depth_queue_.resize(queue_size_);
      depth_queue_.push_back(image_mat);
      break;
    case CameraLayer::Segmentation:
      if (segmentation_queue_.size() > queue_size_)
        segmentation_queue_.resize(queue_size_);
      segmentation_queue_.push_back(image_mat);
      break;
    case CameraLayer::OpticalFlow:
      if (opticalflow_queue_.size() > queue_size_)
        opticalflow_queue_.resize(queue_size_);
      opticalflow_queue_.push_back(image_mat);
      break;
  }
  queue_mutex_.unlock();
  return true;
}

bool RGBCamera::setRelPose(const Ref<Vector<3>> B_r_BC,
                           const Ref<Matrix<3, 3>> R_BC) {
  if (!B_r_BC.allFinite() || !R_BC.allFinite()) {
    logger_.error(
      "The setting value for Camera Relative Pose Matrix is not valid, discard "
      "the setting.");
    return false;
  }
  B_r_BC_ = B_r_BC;
  T_BC_.block<3, 3>(0, 0) = R_BC;
  T_BC_.block<3, 1>(0, 3) = B_r_BC;
  T_BC_.row(3) << 0.0, 0.0, 0.0, 1.0;
  return true;
}

bool RGBCamera::setWidth(const int width) {
  if (width <= 0) {
    logger_.warn(
      "The setting value for Image Width is not valid, discard the setting.");
    return false;
  }
  width_ = width;
  updateCameraIntrinsics();
  return true;
}

bool RGBCamera::setHeight(const int height) {
  if (height <= 0) {
    logger_.warn(
      "The setting value for Image Height is not valid, discard the "
      "setting.");
    return false;
  }
  height_ = height;
  updateCameraIntrinsics();
  return true;
}

bool RGBCamera::updateCameraIntrinsics(void) {
  K_(0, 0) = (height_ / 2) / std::tan(M_PI * fov_ / 180.0 / 2.0);
  K_(1, 1) = K_(0, 0);
  K_(0, 2) = (int)width_ / 2;
  K_(1, 2) = (int)height_ / 2;
  K_(2, 2) = 1;
  return true;
}

bool RGBCamera::setChannels(const int channels) {
  if (!(channels == 1 || channels == 2 || channels == 3)) {
    logger_.warn(
      "The setting value for Image channel is not valid, discard the setting. "
      "%d",
      channels);
    return false;
  }
  channels_ = channels;
  return true;
}

bool RGBCamera::setFOV(const Scalar fov) {
  if (fov <= 0.0) {
    logger_.warn(
      "The setting value for Camera Field-of-View is not valid, discard the "
      "setting.");
    return false;
  }
  fov_ = fov;
  updateCameraIntrinsics();
  return true;
}

bool RGBCamera::setDepthScale(const Scalar depth_scale) {
  if (depth_scale_ < 0.0 || depth_scale_ > 1.0) {
    logger_.warn(
      "The setting value for Camera Depth Scale is not valid, discard the "
      "setting.");
    return false;
  }
  depth_scale_ = depth_scale;
  return true;
}

bool RGBCamera::setPostProcessing(const std::vector<bool>& enabled_layers) {
  if (enabled_layers_.size() != enabled_layers.size()) {
    logger_.warn(
      "Vector size does not match. The vector size should be equal to %d.",
      enabled_layers_.size());
    return false;
  }
  enabled_layers_ = enabled_layers;
  return true;
}

Vector<2> RGBCamera::projectPointToImage(const Ref<Matrix<4, 4>> T_BW,
                                         const Ref<Vector<3>> point_W,
                                         const bool normalized) {
  if (!T_BW.allFinite()) {
    logger_.error(
      "Transformation matrix is not valid. Cannot project point to the image "
      "plane");
  }
  // transformation matrix from body frame to the camera frame
  Matrix<4, 4> T_BC = T_BC_;
  T_BC.block<3, 3>(0, 0) =
    (AngleAxis(-90.0 * M_PI / 180.0, Vector<3>::UnitZ()) *
     AngleAxis(-90 * M_PI / 180.0, Vector<3>::UnitX()))
      .toRotationMatrix();
  Matrix<4, 4> T_CB = inversePoseMatrix(T_BC);

  // transfer point in the world frame to the camera frame
  const Vector<4> point_C =
    T_CB * T_BW * (Vector<4>() << point_W, 1.0).finished();

  //
  Vector<2> point_uv;
  point_uv[0] = (K_(0, 0) * point_C[0]) / point_C[2] + K_(0, 2);
  point_uv[1] = (K_(1, 1) * point_C[1]) / point_C[2] + K_(1, 2);

  if (normalized) {
    point_uv[0] = 2.0 * (point_uv[0] - (Scalar)width_ / 2.0) / (Scalar)width_;
    point_uv[1] = 2.0 * (point_uv[1] - (Scalar)height_ / 2.0) / (Scalar)height_;
    point_uv = point_uv.cwiseMax(-1.0).cwiseMin(1.0);
  }
  return point_uv;
}

std::vector<bool> RGBCamera::getEnabledLayers(void) const {
  return enabled_layers_;
}

Matrix<4, 4> RGBCamera::getRelPose(void) const { return T_BC_; }

int RGBCamera::getChannels(void) const { return channels_; }

int RGBCamera::getWidth(void) const { return width_; }

int RGBCamera::getHeight(void) const { return height_; }

Scalar RGBCamera::getFOV(void) const { return fov_; }

Scalar RGBCamera::getDepthScale(void) const { return depth_scale_; }

Matrix<3, 3> RGBCamera::getIntrinsic(void) const { return K_; }

void RGBCamera::enableDepth(const bool on) {
  if (enabled_layers_[CameraLayer::DepthMap] == on) {
    logger_.warn("Depth layer was already %s.", on ? "on" : "off");
  }
  enabled_layers_[CameraLayer::DepthMap] = on;
}

void RGBCamera::enableSegmentation(const bool on) {
  if (enabled_layers_[CameraLayer::Segmentation] == on) {
    logger_.warn("Segmentation layer was already %s.", on ? "on" : "off");
  }
  enabled_layers_[CameraLayer::Segmentation] = on;
}

void RGBCamera::enableOpticalFlow(const bool on) {
  if (enabled_layers_[CameraLayer::OpticalFlow] == on) {
    logger_.warn("Optical Flow layer was already %s.", on ? "on" : "off");
  }
  enabled_layers_[CameraLayer::OpticalFlow] = on;
}

bool RGBCamera::getRGBImage(cv::Mat& rgb_img) {
  if (!rgb_queue_.empty()) {
    rgb_img = rgb_queue_.front();
    rgb_queue_.pop_front();
    return true;
  }
  return false;
}

bool RGBCamera::getDepthMap(cv::Mat& depth_map) {
  if (!depth_queue_.empty()) {
    cv::Mat ground_truth_depth = depth_queue_.front() * 100.0;
    depth_queue_.pop_front();

    if (depth_noise_fused_) {
      depth_map = fuseDepthWithUncertainty(ground_truth_depth);
    } else {
      depth_map = ground_truth_depth;
    }
    return true;
  }
  return false;
}

bool RGBCamera::getSegmentation(cv::Mat& segmentation) {
  if (!segmentation_queue_.empty()) {
    segmentation = segmentation_queue_.front();
    segmentation_queue_.pop_front();
    return true;
  }
  return false;
}

bool RGBCamera::getOpticalFlow(cv::Mat& opticalflow) {
  if (!opticalflow_queue_.empty()) {
    opticalflow = opticalflow_queue_.front();
    opticalflow_queue_.pop_front();
    return true;
  }
  return false;
}

void RGBCamera::setDepthUncertaintyCoeffs(const std::vector<Scalar>& coeffs) {
  cov_coeffs_ = coeffs;
}

cv::Mat RGBCamera::fuseDepthWithUncertainty(const cv::Mat& depth_map) const {
  if (!depth_noise_fused_) {
    return depth_map;
  }

  cv::Mat fused_depth = depth_map.clone();
  std::random_device rd;
  std::mt19937 gen(rd());

  for (uint16_t y = 0; y < fused_depth.rows; ++y) {
    for (uint16_t x = 0; x < fused_depth.cols; ++x) {
      float z = fused_depth.at<float>(y, x);
      if (z < 1 || z > 10) continue;

      Eigen::Vector3d point_3d((x - K_(0, 2)) * z / K_(0, 0),
                               (y - K_(1, 2)) * z / K_(1, 1),
                               z);

      Eigen::Vector3d cov_diag = getCovariance(point_3d);

      // Generate random noise
      std::normal_distribution<> dist_x(0, std::sqrt(cov_diag.x()));
      std::normal_distribution<> dist_y(0, std::sqrt(cov_diag.y()));
      std::normal_distribution<> dist_z(0, std::sqrt(cov_diag.z()));

      // Add noise to the 3D point
      float fused_z = point_3d.z() + dist_z(gen);
      // Project back to depth frame
      float fused_x = point_3d.x() + dist_x(gen);
      float fused_y = point_3d.y() + dist_y(gen);

      // Check if the projected point is within the image bounds
      uint16_t projected_fused_x = static_cast<uint16_t>(fused_x * K_(0, 0) / fused_z + K_(0, 2));
      uint16_t projected_fused_y = static_cast<uint16_t>(fused_y * K_(1, 1) / fused_z + K_(1, 2));
      if (projected_fused_x < fused_depth.cols && projected_fused_y < fused_depth.rows) {
        // Clip to valid range and convert to 16-bit
        fused_depth.at<float>(projected_fused_y, projected_fused_x) = fused_z;
      }
    }
  }
  return fused_depth;
}

Eigen::Vector3d RGBCamera::getCovariance(const Eigen::Vector3d& depth_point) const {
  double ca0 = cov_coeffs_[0];
  double ca1 = cov_coeffs_[1];
  double ca2 = cov_coeffs_[2];
  double cl0 = cov_coeffs_[3];
  double cl1 = cov_coeffs_[4];
  double cl2 = cov_coeffs_[5];

  double z = depth_point.z();
  double z_2 = z * z;

  Eigen::Vector3d cov;
  cov.x() = ca0 + ca1 * z + ca2 * z_2;
  cov.y() = ca0 + ca1 * z + ca2 * z_2;
  cov.z() = cl0 + cl1 * z + cl2 * z_2;

  return cov;
}

}  // namespace flightlib