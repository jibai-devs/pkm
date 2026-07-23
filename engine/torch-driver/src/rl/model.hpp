// Direct C++ port of MyModel from references/mcts.py.
//
// Encoder: EmbeddingBag(sum) -> TransformerEncoder -> Linear -> tanh(mean)  => value
// Decoder: EmbeddingBag(sum) -> N x DecoderLayer(cross-attn) -> Linear -> tanh => policy
//
// Feature-dimension globals (encoder_size / decoder_size / card_count /
// attack_count) are resolved from the engine's card data at startup and passed
// into the model, mirroring the module-level globals in the Python reference.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace rl {

// Fixed feature layout constants (identical to references/mcts.py).
inline constexpr int64_t kNumWordsEncoder = 24;
inline constexpr int64_t kEncoderSize = 22000; // > vocabulary size
inline constexpr int64_t kDecoderMainFeature = 8; // SelectContext.Main count
inline constexpr int64_t kDecoderAttackOffset = 14; // first attack feature index

// Runtime-resolved dims (set once from engine card/attack data).
struct FeatureDims {
  int64_t card_count = 0;
  int64_t attack_count = 0;
  // decoder_card_offset = decoder_attack_offset + attack_count
  int64_t decoder_card_offset() const { return kDecoderAttackOffset + attack_count; }
  // decoder_size = decoder_card_offset + (1 + main + RECOVER_SPECIAL_CONDITION)*card_count
  int64_t decoder_size(int64_t recover_special_condition) const {
    return decoder_card_offset() +
           (1 + kDecoderMainFeature + recover_special_condition) * card_count;
  }
};

// Decoder layer: cross-attention block (references/mcts.py DecoderLayer).
class DecoderLayerImpl : public torch::nn::Module {
 public:
  DecoderLayerImpl(int64_t d_model, int64_t num_heads, int64_t d_ff)
      : attention(torch::nn::MultiheadAttentionOptions(d_model, num_heads)),
        fc1(d_model, d_ff),
        fc2(d_ff, d_model),
        norm1(torch::nn::LayerNormOptions({d_model})),
        norm2(torch::nn::LayerNormOptions({d_model})) {
    register_module("attention", attention);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("norm1", norm1);
    register_module("norm2", norm2);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_out) {
    // need_weights=false -> ignore returned attention weights. An explicit
    // empty Tensor is required for key_padding_mask (a braced {} can't deduce).
    auto y = std::get<0>(attention(x, encoder_out, encoder_out,
                                   /*key_padding_mask=*/torch::Tensor(),
                                   /*need_weights=*/false));
    auto res = norm1(x + y);
    y = fc1(res);
    y = torch::relu(y);
    y = fc2(y);
    return norm2(res + y);
  }

  torch::nn::MultiheadAttention attention;
  torch::nn::Linear fc1, fc2;
  torch::nn::LayerNorm norm1, norm2;
};
TORCH_MODULE(DecoderLayer);

class MyModelImpl : public torch::nn::Module {
 public:
  MyModelImpl(int64_t d_model, int64_t num_heads, int64_t d_ff,
              int64_t num_layers_encoder, int64_t num_layers_decoder,
              int64_t decoder_size)
      : d_model_(d_model),
        encoder_bag(torch::nn::EmbeddingBagOptions(kEncoderSize, d_model)
                        .mode(torch::kSum)),
        encoder(torch::nn::TransformerEncoderLayer(
                    torch::nn::TransformerEncoderLayerOptions(d_model, num_heads)
                        .dim_feedforward(d_ff)
                        .dropout(0)),
                num_layers_encoder),
        encoder_fc(d_model, 1),
        decoder_bag(torch::nn::EmbeddingBagOptions(decoder_size, d_model)
                        .mode(torch::kSum)),
        decoder_fc(d_model, 1) {
    register_module("encoder_bag", encoder_bag);
    register_module("encoder", encoder);
    register_module("encoder_fc", encoder_fc);
    register_module("decoder_bag", decoder_bag);
    decoder = register_module("decoder", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers_decoder; ++i) {
      auto layer = DecoderLayer(d_model, num_heads, d_ff);
      decoder->push_back(layer);
      decoder_layers.push_back(layer);
    }
    register_module("decoder_fc", decoder_fc);
  }

  // Returns (value, policy). EmbeddingBag inputs are (index, value, offset).
  std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor index_encoder, torch::Tensor value_encoder,
      torch::Tensor offset_encoder, torch::Tensor index_decoder,
      torch::Tensor value_decoder, torch::Tensor offset_decoder) {
    // EmbeddingBag(input, offsets, per_sample_weights)
    auto v = encoder_bag(index_encoder, offset_encoder, value_encoder);
    v = v.reshape({-1, kNumWordsEncoder, d_model_}).transpose(0, 1);
    int64_t batch_size = v.size(1);
    auto encoder_out = encoder(v);
    v = encoder_fc(encoder_out);
    v = torch::tanh(v.mean(0));

    auto p = decoder_bag(index_decoder, offset_decoder, value_decoder);
    p = p.reshape({batch_size, -1, d_model_}).transpose(0, 1);
    for (auto& layer : decoder_layers) {
      p = layer->forward(p, encoder_out);
    }
    p = decoder_fc(p);
    p = p.transpose(0, 1).view({batch_size, -1});
    p = torch::tanh(p);
    return {v, p};
  }

  int64_t d_model_;
  torch::nn::EmbeddingBag encoder_bag;
  torch::nn::TransformerEncoder encoder;
  torch::nn::Linear encoder_fc;
  torch::nn::EmbeddingBag decoder_bag;
  torch::nn::ModuleList decoder;
  std::vector<DecoderLayer> decoder_layers;
  torch::nn::Linear decoder_fc;
};
TORCH_MODULE(MyModel);

}  // namespace rl
