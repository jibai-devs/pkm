// Toolchain proof for the C++ libtorch driver.
//
// This target is built under a GCC/libstdc++ stdenv (matching libtorch-bin's
// ABI), NOT the libc++ stdenv used for cg.so. It talks to the engine only
// through its extern "C" ABI (POD structs + pointers), so no STL crosses the
// boundary and there is no libc++/libstdc++ collision.
#include <torch/torch.h>

#include <cstdio>
#include <dlfcn.h>

int main(int argc, char **argv) {
  // 1. libtorch is linked and CUDA is reachable (the whole point: the 3090).
  std::printf("torch %s\n", TORCH_VERSION);
  const bool cuda = torch::cuda::is_available();
  std::printf("cuda available: %s (%d device(s))\n", cuda ? "yes" : "no",
              cuda ? static_cast<int>(torch::cuda::device_count()) : 0);

  auto dev = cuda ? torch::kCUDA : torch::kCPU;
  auto a = torch::randn({2, 3}, torch::device(dev));
  auto b = torch::randn({3, 2}, torch::device(dev));
  auto c = torch::mm(a, b);
  std::printf("matmul ok, result on %s\n", cuda ? "cuda" : "cpu");
  (void)c;

  // 2. The engine's C ABI is callable from this GCC-built binary.
  //    Path to cg.so is arg 1 (defaults to the vendored build output).
  const char *lib = argc > 1 ? argv[1] : "../build/cg.so";
  void *h = dlopen(lib, RTLD_NOW);
  if (!h) {
    std::printf("dlopen(%s) failed: %s\n", lib, dlerror());
    return 1;
  }
  auto GameInitialize = reinterpret_cast<void (*)()>(dlsym(h, "GameInitialize"));
  if (!GameInitialize) {
    std::printf("dlsym(GameInitialize) failed: %s\n", dlerror());
    return 1;
  }
  GameInitialize();
  std::printf("cg.so C ABI call ok (GameInitialize)\n");
  dlclose(h);
  return 0;
}
