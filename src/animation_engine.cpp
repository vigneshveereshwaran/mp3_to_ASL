#include "animation_engine.h"
#include <cstring>
#include <iostream>

namespace hearlink {

AnimationEngine::AnimationEngine()
    : transition_duration_(0.15f), blend_progress_(1.0f) {
    current_pose_.resize(67); // 25 upper body + 21 right hand + 21 left hand
    target_pose_.resize(67);
}

AnimationEngine::~AnimationEngine() {}

void AnimationEngine::set_transition_duration(float seconds) {
    transition_duration_ = std::max(0.01f, seconds);
}

void AnimationEngine::apply_ccd_ik(Vec3* bone_chain, int chain_len, Vec3 target, int max_iters) {
    if (chain_len < 2) return;

    for (int iter = 0; iter < max_iters; ++iter) {
        Vec3 end_effector = bone_chain[chain_len - 1];
        if ((end_effector - target).length_sq() < 1e-4f) break;

        for (int i = chain_len - 2; i >= 0; --i) {
            Vec3 cur_joint = bone_chain[i];
            Vec3 to_end = (bone_chain[chain_len - 1] - cur_joint).normalized();
            Vec3 to_target = (target - cur_joint).normalized();

            Quaternion rot = Quaternion::from_to_rotation(to_end, to_target);

            // Apply rotation to remaining joints in chain
            for (int j = i + 1; j < chain_len; ++j) {
                Vec3 dir = bone_chain[j] - cur_joint;
                // Rotate vector dir by rot
                Vec3 qv(rot.x, rot.y, rot.z);
                Vec3 uv = Vec3::cross(qv, dir);
                Vec3 uuv = Vec3::cross(qv, uv);
                dir = dir + (uv * (2.0f * rot.w)) + (uuv * 2.0f);
                bone_chain[j] = cur_joint + dir;
            }
        }
    }
}

void AnimationEngine::process_frame(const float* raw_keypoints, int num_landmarks, float dt, float* output_transforms) {
    // Process input keypoints into quaternions and position vectors
    int landmarks_count = std::min(num_landmarks, 67);

    if (dt > 0.0001f && blend_progress_ < 1.0f) {
        blend_progress_ += dt / transition_duration_;
        blend_progress_ = std::min(1.0f, blend_progress_);
    } else {
        blend_progress_ = 1.0f;
    }

    for (int i = 0; i < landmarks_count; ++i) {
        Vec3 keypoint(raw_keypoints[i * 3], raw_keypoints[i * 3 + 1], raw_keypoints[i * 3 + 2]);

        // Compute local joint rotation relative to reference up vector (0, 1, 0)
        Quaternion rot = Quaternion::from_to_rotation(Vec3(0.0f, 1.0f, 0.0f), keypoint.normalized());
        target_pose_[i].rotation = rot;
        target_pose_[i].position = keypoint;

        if (blend_progress_ >= 1.0f) {
            current_pose_[i] = target_pose_[i];
        } else {
            current_pose_[i].rotation = Quaternion::slerp(current_pose_[i].rotation, target_pose_[i].rotation, blend_progress_);
            current_pose_[i].position = Vec3::lerp(current_pose_[i].position, target_pose_[i].position, blend_progress_);
        }

        // Write output: 4 floats rotation + 3 floats position = 7 floats per bone
        int out_idx = i * 7;
        output_transforms[out_idx + 0] = current_pose_[i].rotation.x;
        output_transforms[out_idx + 1] = current_pose_[i].rotation.y;
        output_transforms[out_idx + 2] = current_pose_[i].rotation.z;
        output_transforms[out_idx + 3] = current_pose_[i].rotation.w;
        output_transforms[out_idx + 4] = current_pose_[i].position.x;
        output_transforms[out_idx + 5] = current_pose_[i].position.y;
        output_transforms[out_idx + 6] = current_pose_[i].position.z;
    }
}

} // namespace hearlink

extern "C" {

EMSCRIPTEN_KEEPALIVE float* create_animation_engine() {
    hearlink::AnimationEngine* engine = new hearlink::AnimationEngine();
    return reinterpret_cast<float*>(engine);
}

EMSCRIPTEN_KEEPALIVE void destroy_animation_engine(float* engine_ptr) {
    if (engine_ptr) {
        hearlink::AnimationEngine* engine = reinterpret_cast<hearlink::AnimationEngine*>(engine_ptr);
        delete engine;
    }
}

EMSCRIPTEN_KEEPALIVE void process_pose_frame_wasm(float* engine_ptr, const float* input_buf, int num_landmarks, float dt, float* output_buf) {
    if (engine_ptr && input_buf && output_buf) {
        hearlink::AnimationEngine* engine = reinterpret_cast<hearlink::AnimationEngine*>(engine_ptr);
        engine->process_frame(input_buf, num_landmarks, dt, output_buf);
    }
}

EMSCRIPTEN_KEEPALIVE void slerp_quaternion_wasm(float q1x, float q1y, float q1z, float q1w,
                                                float q2x, float q2y, float q2z, float q2w,
                                                float t, float* out_q) {
    hearlink::Quaternion q1(q1x, q1y, q1z, q1w);
    hearlink::Quaternion q2(q2x, q2y, q2z, q2w);
    hearlink::Quaternion res = hearlink::Quaternion::slerp(q1, q2, t);
    out_q[0] = res.x;
    out_q[1] = res.y;
    out_q[2] = res.z;
    out_q[3] = res.w;
}

}
