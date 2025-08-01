//
// Created by biao on 24-10-6.
//

#include "ObservationBuffer.h"

/**
 * @brief Construct a new Observation Buffer object
 * 
 * @param num_envs 表示环境数量
 * @param num_obs 表示每个环境的观测维度
 * @param include_history_steps 表示历史观测步数
 */
ObservationBuffer::ObservationBuffer(int num_envs,
                                     const int num_obs,
                                     const int include_history_steps)
    : num_envs_(num_envs),
      num_obs_(num_obs),
      include_history_steps_(include_history_steps) {
    num_obs_total_ = num_obs_ * include_history_steps_;
    // 初始化观测缓存，形状为(num_envs, num_obs_total_)，数据类型为float32，并初始化为0
    obs_buffer_ = torch::zeros({num_envs_, num_obs_total_}, dtype(torch::kFloat32));
}
/**
 * @brief 重置观测缓存
 * 
 * @param reset_index 表示需要重置的环境索引
 * @param new_obs 表示新的观测值，形状为(num_envs, num_obs)
 */
void ObservationBuffer::reset(const std::vector<int> &reset_index, const torch::Tensor &new_obs) {
    std::vector<torch::indexing::TensorIndex> indices;
    for (int index: reset_index) {
        indices.emplace_back(torch::indexing::Slice(index));
    }
    obs_buffer_.index_put_(indices, new_obs.repeat({1, include_history_steps_}));
}

/**
 * @brief 清空观测缓存，将缓存中的所有元素置为 0
 */
void ObservationBuffer::clear()
{
    // 创建一个与 obs_buffer_ 形状和数据类型相同，但所有元素都为 0 的张量，并赋值给 obs_buffer_
    obs_buffer_ = torch::zeros_like(obs_buffer_);
}

/**
 * @brief 向观测缓存中插入新的观测值，同时将原有观测值向后移动
 * 
 * @param new_obs 新的观测值，形状为 (num_envs, num_obs)
 */
void ObservationBuffer::insert(const torch::Tensor &new_obs) {
    // Shift observations back.
    // 选取 obs_buffer_ 中每个环境从第 num_obs_ 个元素开始到最后的部分
    // torch::indexing::Slice(torch::indexing::None) 表示在第一个维度（环境维度）选取所有元素
    // torch::indexing::Slice(num_obs_, num_obs_ * include_history_steps_) 表示在第二个维度（观测维度）选取从 num_obs_ 到 num_obs_ * include_history_steps_ 的元素
    // 并对选取的部分进行深拷贝，得到新的张量 shifted_obs
    const torch::Tensor shifted_obs = obs_buffer_.index({
        torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(num_obs_, num_obs_ * include_history_steps_)
    }).clone();

    // 将 shifted_obs 赋值给 obs_buffer_ 中每个观测维度的前 num_obs_ * (include_history_steps_ - 1) 个元素
    // 实现观测值向后移动
    obs_buffer_.index({
        torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, num_obs_ * (include_history_steps_ - 1))
    }) = shifted_obs;

    // Add new observation.
    obs_buffer_.index({
        torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-num_obs_, torch::indexing::None)
    }) = new_obs;
}

torch::Tensor ObservationBuffer::getObsVec(const std::vector<int> &obs_ids) const {
    std::vector<torch::Tensor> obs;
    for (int i = obs_ids.size() - 1; i >= 0; --i) {
        const int obs_id = obs_ids[i];
        const int slice_idx = include_history_steps_ - obs_id - 1;
        obs.push_back(obs_buffer_.index({
            torch::indexing::Slice(torch::indexing::None),
            torch::indexing::Slice(slice_idx * num_obs_, (slice_idx + 1) * num_obs_)
        }));
    }
    return cat(obs, -1);
}
