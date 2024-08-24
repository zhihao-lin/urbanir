#include "utils.h"


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(half_sizes);
    return ray_aabb_intersect_cu(rays_o, rays_d, centers, half_sizes, max_hits);
}


std::vector<torch::Tensor> ray_sphere_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    return ray_sphere_intersect_cu(rays_o, rays_d, centers, radii, max_hits);
}


void packbits(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
){
    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);
    
    return packbits_cu(density_grid, density_threshold, density_bitfield);
}


torch::Tensor morton3D(const torch::Tensor coords){
    CHECK_INPUT(coords);

    return morton3D_cu(coords);
}


torch::Tensor morton3D_invert(const torch::Tensor indices){
    CHECK_INPUT(indices);

    return morton3D_invert_cu(indices);
}


std::vector<torch::Tensor> raymarching_train(
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
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(density_bitfield);
    CHECK_INPUT(noise);

    return raymarching_train_cu(
        rays_o, rays_d, hits_t, density_bitfield, cascades,
        scale, exp_step_factor, noise, grid_size, max_samples);
}


std::vector<torch::Tensor> raymarching_test(
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
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(density_bitfield);

    return raymarching_test_cu(
        rays_o, rays_d, hits_t, alive_indices, density_bitfield, cascades,
        scale, exp_step_factor, grid_size, max_samples, N_samples);
}

std::vector<torch::Tensor> composite_alpha_fw(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);

    return composite_alpha_fw_cu(sigmas, deltas, rays_a, opacity_threshold);
}

std::vector<torch::Tensor> composite_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor albedos,
    const torch::Tensor normals_pred,
    const torch::Tensor sems,
    const torch::Tensor viss,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(albedos);
    CHECK_INPUT(normals_pred);
    CHECK_INPUT(sems);
    CHECK_INPUT(viss);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_train_fw_cu(
                sigmas, rgbs, albedos, normals_pred, sems, viss, deltas, ts,
                rays_a, opacity_threshold, classes);
}

std::vector<torch::Tensor> composite_train_bw(
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
    const float opacity_threshold,
    const int classes
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_ddepth);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(dL_dalbedo);
    CHECK_INPUT(dL_dnormal_pred);
    CHECK_INPUT(dL_dsem);
    CHECK_INPUT(dL_dvis);
    CHECK_INPUT(dL_dws);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(albedos);
    CHECK_INPUT(viss);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    CHECK_INPUT(albedo);
    CHECK_INPUT(vis);

    return composite_train_bw_cu(
                dL_dopacity, dL_ddepth, dL_drgb, dL_dalbedo, dL_dnormal_pred, dL_dsem, dL_dvis, dL_dws,
                sigmas, rgbs, albedos, viss, ws, deltas, ts, rays_a,
                opacity, depth, rgb, albedo, vis, opacity_threshold, classes);
}
////////////////////////////////////////////////
std::vector<torch::Tensor> composite_refloss_fw(
    const torch::Tensor sigmas,
    const torch::Tensor normals_diff,
    const torch::Tensor normals_ori,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(normals_diff);
    CHECK_INPUT(normals_ori);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_refloss_fw_cu(
                sigmas, normals_diff, normals_ori, deltas, ts,
                rays_a, opacity_threshold);
}


std::vector<torch::Tensor> composite_refloss_bw(
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
){
    CHECK_INPUT(dL_dloss_o);
    CHECK_INPUT(dL_dloss_p);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(normals_diff);
    CHECK_INPUT(normals_ori);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(loss_o);
    CHECK_INPUT(loss_p);

    return composite_refloss_bw_cu(
                dL_dloss_o, dL_dloss_p,
                sigmas, normals_diff, normals_ori, deltas, ts, rays_a,
                loss_o, loss_p, opacity_threshold);
}
////////////////////////////////////////////////
void composite_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor albedos,
    const torch::Tensor normals,
    const torch::Tensor normals_raw,
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
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(albedos);
    CHECK_INPUT(normals);
    CHECK_INPUT(normals_raw);
    CHECK_INPUT(sems);
    CHECK_INPUT(viss);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);
    CHECK_INPUT(albedo);
    CHECK_INPUT(normal);
    CHECK_INPUT(normal_raw);
    CHECK_INPUT(sem);
    CHECK_INPUT(vis);

    composite_test_fw_cu(
        sigmas, rgbs, albedos, normals, normals_raw, sems, viss,
        deltas, ts, hits_t, alive_indices,
        T_threshold, classes, N_eff_samples,
        opacity, depth, rgb, albedo, normal, normal_raw, sem, vis);
}

torch::Tensor composite_weight_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    const torch::Tensor opacity
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(deltas);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);

    return composite_weight_test_fw_cu(
        sigmas, deltas, alive_indices, T_threshold,
        N_eff_samples, opacity 
    );
}

void composite_visibility_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor viss_t,
    const torch::Tensor deltas,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor visibility_T
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(viss_t);
    CHECK_INPUT(deltas);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);
    CHECK_INPUT(visibility_T);

    composite_visibility_test_fw_cu(
        sigmas, viss_t, deltas, 
        alive_indices, T_threshold, 
        N_eff_samples, opacity, visibility_T);
}

std::vector<torch::Tensor> visibility_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor light_dist,
    const float T_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(light_dist);

    return visibility_train_fw_cu(
        sigmas, deltas, ts, rays_a, light_dist, T_threshold
    );
}

torch::Tensor visibility_train_bw(
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
){
    CHECK_INPUT(dL_dvisibility);
    CHECK_INPUT(dL_dws);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(visibility);
    CHECK_INPUT(light_dist);

    return visibility_train_bw_cu(
        dL_dvisibility, dL_dws, sigmas, ws, deltas, ts,
        rays_a, visibility, light_dist, T_threshold
    );
}

void visibility_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    const torch::Tensor light_dist,
    torch::Tensor visibility
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(light_dist);
    CHECK_INPUT(visibility);

    visibility_test_fw_cu(
        sigmas, deltas, ts, alive_indices, T_threshold, 
        N_eff_samples, light_dist, visibility
    );
}

std::vector<torch::Tensor> distortion_loss_fw(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return distortion_loss_fw_cu(ws, deltas, ts, rays_a);
}


torch::Tensor distortion_loss_bw(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    CHECK_INPUT(dL_dloss);
    CHECK_INPUT(ws_inclusive_scan);
    CHECK_INPUT(wts_inclusive_scan);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return distortion_loss_bw_cu(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                 ws, deltas, ts, rays_a);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_sphere_intersect", &ray_sphere_intersect);

    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("packbits", &packbits);

    m.def("raymarching_train", &raymarching_train);
    m.def("raymarching_test", &raymarching_test);
    m.def("composite_alpha_fw", &composite_alpha_fw);
    m.def("composite_train_fw", &composite_train_fw);
    m.def("composite_train_bw", &composite_train_bw);
    m.def("composite_refloss_fw", &composite_refloss_fw);
    m.def("composite_refloss_bw", &composite_refloss_bw);
    m.def("composite_test_fw", &composite_test_fw);
    m.def("composite_weight_test_fw", &composite_weight_test_fw);
    m.def("composite_visibility_test_fw", &composite_visibility_test_fw);

    m.def("visibility_train_fw", &visibility_train_fw);
    m.def("visibility_train_bw", &visibility_train_bw);
    m.def("visibility_test_fw", &visibility_test_fw);

    m.def("distortion_loss_fw", &distortion_loss_fw);
    m.def("distortion_loss_bw", &distortion_loss_bw);
}