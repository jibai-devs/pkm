// Self-play RL trainer — direct C++ port of the __main__ loop in
// references/mcts.py. Evaluate vs random, collect self-play data with MCTS,
// compute TD(lambda) value targets, train the Transformer with masked Huber
// losses. Runs against the engine (cg.so) via the C ABI, uses libtorch for the
// net, and the 3090 when the CUDA build is used.
#include <torch/torch.h>

#include <CLI/CLI.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "engine.hpp"
#include "features.hpp"
#include "mcts.hpp"
#include "model.hpp"
#include "obs.hpp"
#include "sparse.hpp"

namespace {

using Clock = std::chrono::steady_clock;
inline double secs_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}


// A sample deck for training (identical to references/mcts.py sample_deck).
const std::vector<int> kSampleDeck = {
    721,  721,  722,  722,  722,  722,  723,  723,  723,  723,  1092, 1121, 1121,
    1145, 1145, 1163, 1163, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1262,
    1262, 3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,    3,
    3,    3,    3,    3,    3,    3,    3,    3};

constexpr int kBatchSize = 128;
constexpr int kMaxWords = 64;  // decoder words padded per sample
constexpr double kLambda = 0.9;

// Random opponent: pick maxCount distinct option indices at random.
std::vector<int> random_agent(const rl::Observation& obs) {
  std::vector<int> range(obs.select.option.size());
  for (size_t i = 0; i < range.size(); ++i) range[i] = static_cast<int>(i);
  return rl::sample_k(range, obs.select.maxCount);
}

}  // namespace

int main(int argc, char** argv) {
  // --- CLI (CLI11) ----------------------------------------------------------
  CLI::App app{"pkm C++ MCTS self-play trainer (port of references/mcts.py)"};
  int iterations = 5, eval_games = 50, selfplay_games = 100;
  int search_count = 10;
  int d_model = 128, n_heads = 2, d_ff = 256, enc_layers = 1, dec_layers = 1;
  double lr = 3e-4;
  std::string device_arg = "auto";
  std::string out_dir = "out";
  app.add_option("--iters", iterations, "Training iterations")->capture_default_str();
  app.add_option("--eval-games", eval_games, "Eval games vs random per iter")
      ->capture_default_str();
  app.add_option("--selfplay-games", selfplay_games, "Self-play games per iter")
      ->capture_default_str();
  app.add_option("--search-count", search_count, "MCTS simulations per decision")
      ->capture_default_str();
  app.add_option("--device", device_arg, "auto|cpu|cuda")
      ->check(CLI::IsMember({"auto", "cpu", "cuda"}))
      ->capture_default_str();
  app.add_option("--lr", lr, "AdamW learning rate")->capture_default_str();
  app.add_option("--d-model", d_model, "Model width")->capture_default_str();
  app.add_option("--n-heads", n_heads, "Attention heads")->capture_default_str();
  app.add_option("--d-ff", d_ff, "Feedforward dim")->capture_default_str();
  app.add_option("--enc-layers", enc_layers, "Encoder layers")->capture_default_str();
  app.add_option("--dec-layers", dec_layers, "Decoder layers")->capture_default_str();
  app.add_option("--out", out_dir, "Output dir for checkpoints")->capture_default_str();
  CLI11_PARSE(app, argc, argv);
  rl::g_search_count = search_count;

  rl::engine::Engine eng;

  // --- resolve feature dimensions from engine card data --------------------
  auto cards = rl::engine::all_cards();
  auto attacks = rl::engine::all_attacks();
  int64_t card_count = 0, attack_count = 0;
  for (const auto& c : cards)
    card_count = std::max<int64_t>(card_count, c.value("cardId", 0) + 1);
  for (const auto& a : attacks)
    attack_count = std::max<int64_t>(attack_count, a.value("attackId", 0) + 1);
  rl::init_dims(card_count, attack_count);
  const int64_t decoder_size = rl::g_dims.decoder_size(rl::kRecoverSpecialCondition);
  std::printf("cards=%lld attacks=%lld decoder_size=%lld\n",
              (long long)card_count, (long long)attack_count,
              (long long)decoder_size);

  // Device: honor --device; "auto" falls back to CPU if CUDA is unavailable.
  bool want_cuda = device_arg == "cuda" ||
                   (device_arg == "auto" && torch::cuda::is_available());
  if (want_cuda && !torch::cuda::is_available()) {
    std::printf("WARNING: --device cuda requested but CUDA is unavailable; using CPU\n");
    want_cuda = false;
  }
  torch::Device device(want_cuda ? torch::kCUDA : torch::kCPU);
  std::printf("device: %s\n", device.is_cuda() ? "cuda" : "cpu");

  rl::MyModel model(d_model, n_heads, d_ff, enc_layers, dec_layers, decoder_size);
  model->to(device);
  torch::optim::AdamW optimizer(model->parameters(),
                                torch::optim::AdamWOptions(lr));
  torch::nn::HuberLoss loss_enc(torch::nn::HuberLossOptions().delta(0.2));
  namespace F = torch::nn::functional;

  std::filesystem::create_directories(out_dir);

  std::printf("config: iters=%d eval_games=%d selfplay_games=%d search_count=%d "
              "d_model=%d heads=%d d_ff=%d enc=%d dec=%d lr=%.1e out=%s\n",
              iterations, eval_games, selfplay_games, rl::g_search_count, d_model,
              n_heads, d_ff, enc_layers, dec_layers, lr, out_dir.c_str());
  std::fflush(stdout);

  const auto t_start = Clock::now();
  for (int counter = 0; counter < iterations; ++counter) {
    const auto t_iter = Clock::now();
    torch::save(model, out_dir + "/model" + std::to_string(counter) + ".pt");
    std::vector<rl::LearnSample> sample_list;

    {
      torch::InferenceMode guard;
      model->eval();

      // --- evaluation vs random -------------------------------------------
      int wins = 0, losses = 0, draws = 0;
      long eval_moves = 0, eval_decisions = 0;  // moves = all selects
      const auto t_eval = Clock::now();
      for (int i = 0; i < eval_games; ++i) {
        rl::Observation obs = eng.battle_start(kSampleDeck, kSampleDeck);
        const int your_index = i % 2;
        while (obs.current.result < 0) {
          std::vector<int> selected;
          if (obs.current.yourIndex == your_index) {
            auto [sel, sample] =
                rl::mcts_agent(eng, obs, kSampleDeck, model, device);
            selected = std::move(sel);
            ++eval_decisions;
          } else {
            selected = random_agent(obs);
          }
          obs = eng.battle_select(selected);
          ++eval_moves;
        }
        const int result = obs.current.result;
        eng.battle_finish();
        if (result == 2)
          ++draws;
        else if (result == your_index)
          ++wins;
        else
          ++losses;
        const double el = secs_since(t_eval);
        std::fprintf(stderr, "\rEvaluating... %d%%  (%.2f games/s)   ",
                     100 * (i + 1) / eval_games, (i + 1) / (el > 0 ? el : 1));
      }
      std::fprintf(stderr, "\n");
      const double eval_dt = secs_since(t_eval);
      std::printf("[iter %d] eval: win_rate=%d%% (W%d/L%d/D%d) | %.1fs "
                  "| %.2f games/s, %.1f moves/s, %.1f mcts-decisions/s\n",
                  counter, wins + losses > 0 ? 100 * wins / (wins + losses) : 0,
                  wins, losses, draws, eval_dt, eval_games / eval_dt,
                  eval_moves / eval_dt, eval_decisions / eval_dt);
      std::fflush(stdout);

      // --- self-play data collection --------------------------------------
      long sp_moves = 0;  // every self-play move is an MCTS decision
      const auto t_sp = Clock::now();
      for (int g = 0; g < selfplay_games; ++g) {
        rl::Observation obs = eng.battle_start(kSampleDeck, kSampleDeck);
        std::vector<std::vector<rl::LearnSample>> samples(2);
        while (obs.current.result < 0) {
          const int who = obs.current.yourIndex;
          auto [selected, sample] =
              rl::mcts_agent(eng, obs, kSampleDeck, model, device);
          if (sample) samples[who].push_back(std::move(*sample));
          obs = eng.battle_select(selected);
          ++sp_moves;
        }
        const int result = obs.current.result;
        eng.battle_finish();

        // TD(lambda) value targets, iterating backwards from game end.
        for (int i = 0; i < 2; ++i) {
          double value = (i == result) ? 1.0 : -1.0;
          for (auto it = samples[i].rbegin(); it != samples[i].rend(); ++it) {
            double label = (value + it->value) * 0.5;
            value = value * kLambda + it->value * (1.0 - kLambda);
            it->value = static_cast<float>(label);
            sample_list.push_back(std::move(*it));
          }
        }
        const double el = secs_since(t_sp);
        std::fprintf(stderr,
                     "\rSelf-play... %d%%  (%.2f games/s, %.1f moves/s)   ",
                     100 * (g + 1) / selfplay_games, (g + 1) / (el > 0 ? el : 1),
                     sp_moves / (el > 0 ? el : 1));
      }
      std::fprintf(stderr, "\n");
      const double sp_dt = secs_since(t_sp);
      std::printf("[iter %d] selfplay: %d games | %.1fs | %.2f games/s, "
                  "%.1f moves/s | %zu samples collected\n",
                  counter, selfplay_games, sp_dt, selfplay_games / sp_dt,
                  sp_moves / sp_dt, sample_list.size());
      std::fflush(stdout);
    }

    // --- train on the collected self-play data -----------------------------
    model->train();
    std::shuffle(sample_list.begin(), sample_list.end(), rl::rng());
    const int batch_count = static_cast<int>(sample_list.size()) / kBatchSize;
    const auto t_train = Clock::now();
    double loss_sum = 0.0;
    for (int b = 0; b < batch_count; ++b) {
      rl::LearnInput input_enc, input_dec;
      std::vector<float> mask, label_enc, label_dec;
      const int start = kBatchSize * b;
      for (int j = start; j < start + kBatchSize; ++j) {
        rl::LearnSample& s = sample_list[j];
        input_enc.add(s.sv_enc);
        input_dec.add(s.sv_dec);
        label_enc.push_back(s.value);
        label_dec.insert(label_dec.end(), s.policy.begin(), s.policy.end());
        for (size_t k = 0; k < s.policy.size(); ++k) mask.push_back(1.0f);
        for (int k = 0; k < kMaxWords - (int)s.policy.size(); ++k) {
          mask.push_back(0.0f);
          label_dec.push_back(0.0f);
          input_dec.offset.push_back(static_cast<int32_t>(input_dec.index.size()));
        }
      }

      auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
      auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
      auto mask_t = torch::tensor(mask, f32).view({kBatchSize, -1});
      auto label_enc_t = torch::tensor(label_enc, f32).view({kBatchSize, -1});
      auto label_dec_t = torch::tensor(label_dec, f32).view({kBatchSize, -1});

      optimizer.zero_grad();
      auto [out_enc, out_dec] = model->forward(
          torch::tensor(input_enc.index, i32), torch::tensor(input_enc.value, f32),
          torch::tensor(input_enc.offset, i32), torch::tensor(input_dec.index, i32),
          torch::tensor(input_dec.value, f32), torch::tensor(input_dec.offset, i32));

      auto l_enc = loss_enc(out_enc, label_enc_t);
      auto l_dec = F::huber_loss(
          out_dec, label_dec_t,
          F::HuberLossFuncOptions().delta(0.1).reduction(torch::kNone));
      l_dec = (l_dec * mask_t).sum() / static_cast<float>(kBatchSize);
      auto loss = l_enc + l_dec;
      loss.backward();
      optimizer.step();
      loss_sum += loss.item<double>();
    }
    const double train_dt = secs_since(t_train);
    std::printf("[iter %d] train: %d batches | %.1fs | %.2f batches/s | "
                "mean_loss=%.4f\n",
                counter, batch_count, train_dt,
                batch_count > 0 ? batch_count / train_dt : 0.0,
                batch_count > 0 ? loss_sum / batch_count : 0.0);
    std::printf("[iter %d] DONE in %.1fs (total elapsed %.1fs)\n", counter,
                secs_since(t_iter), secs_since(t_start));
    std::fflush(stdout);
  }
  std::printf("all %d iterations done in %.1fs\n", iterations,
              secs_since(t_start));
  return 0;
}
