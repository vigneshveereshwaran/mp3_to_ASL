#ifndef ANIMATION_ENGINE_H
#define ANIMATION_ENGINE_H

#include "math_utils.h"
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace hearlink {

struct BoneTransform {
    Quaternion rotation;
    Vec3 position;
};

class AnimationEngine {
public:
    AnimationEngine();
    ~AnimationEngine();

    void set_transition_duration(float seconds);
    void process_frame(const float* raw_keypoints, int num_landmarks, float dt, float* output_transforms);
    void apply_ccd_ik(Vec3* bone_chain, int chain_len, Vec3 target, int max_iters = 10);

private:
    float transition_duration_;
    std::vector<BoneTransform> current_pose_;
    std::vector<BoneTransform> target_pose_;
    float blend_progress_;
};

} // namespace hearlink

extern "C" {
    EMSCRIPTEN_KEEPALIVE float* create_animation_engine();
    EMSCRIPTEN_KEEPALIVE void destroy_animation_engine(float* engine_ptr);
    EMSCRIPTEN_KEEPALIVE void process_pose_frame_wasm(float* engine_ptr, const float* input_buf, int num_landmarks, float dt, float* output_buf);
    EMSCRIPTEN_KEEPALIVE void slerp_quaternion_wasm(float q1x, float q1y, float q1z, float q1w, float q2x, float q2y, float q2z, float q2w, float t, float* out_q);
}

#endif // ANIMATION_ENGINE_H
