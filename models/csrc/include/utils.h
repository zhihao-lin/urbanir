#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// #define CHECK_INPUT(x) CHECK_CUDA(x)


std::vector<torch::Tensor> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
);


std::vector<torch::Tensor> ray_sphere_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
);


void packbits_cu(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
);


torch::Tensor morton3D_cu(const torch::Tensor coords);
torch::Tensor morton3D_invert_cu(const torch::Tensor indices);


std::vector<torch::Tensor> raymarching_train_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const torch::Tensor noise,
    const int grid_size,
    const int max_samples
);


std::vector<torch::Tensor> raymarching_test_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples
);

std::vector<torch::Tensor> composite_alpha_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
);

std::vector<torch::Tensor> composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor albedos,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor viss,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float T_threshold,
    const int classes
);


std::vector<torch::Tensor> composite_train_bw_cu(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor dL_dalbedo,
    const torch::Tensor dL_dnormal_pred,
    const torch::Tensor dL_dsem,
    const torch::Tensor dL_dvis,
    const torch::Tensor dL_dws,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor albedos,
    const torch::Tensor viss,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const torch::Tensor albedo,
    const torch::Tensor vis,
    const float T_threshold,
    const int classses
);


std::vector<torch::Tensor> composite_refloss_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor normals_diff,
    const torch::Tensor normals_ori,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold
);


std::vector<torch::Tensor> composite_refloss_bw_cu(
    const torch::Tensor dL_dloss_o,
    const torch::Tensor dL_dloss_p,
    const torch::Tensor sigmas,
    const torch::Tensor normals_diff,
    const torch::Tensor normals_ori,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor loss_o,
    const torch::Tensor loss_p,
    const float opacity_threshold
);

void composite_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor albedos,
    const torch::Tensor noramls,
    const torch::Tensor noramls_raw,
    const torch::Tensor sems,
    const torch::Tensor viss,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
    const float T_threshold,
    const int classes,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb,
    torch::Tensor albedo,
    torch::Tensor normal,
    torch::Tensor normal_raw,
    torch::Tensor sem,
    torch::Tensor vis
);

torch::Tensor composite_weight_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    const torch::Tensor opacity
);

void composite_visibility_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor viss_t,
    const torch::Tensor deltas,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor visibility_T
);

std::vector<torch::Tensor> visibility_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor light_dist,
    const float T_threshold
);

torch::Tensor visibility_train_bw_cu(
    const torch::Tensor dL_dvisibility,
    const torch::Tensor dL_dws,
    const torch::Tensor sigmas,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor visibility,
    const torch::Tensor light_dist,
    const float T_threshold
);

void visibility_test_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    const torch::Tensor light_dist,
    torch::Tensor visibility
);

std::vector<torch::Tensor> distortion_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
);

torch::Tensor distortion_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
);