// PyBind11 wrapper for hip_stock.hip — exposes the launcher to Python via torch's extension API.
//
// load_inline emits one combined .so that needs PYBIND11_MODULE; the kernel itself lives in
// hip_stock.hip. The two files compile separately and link at the .so step.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

extern "C" void rmsnorm_h128_launch(
    const void* x, const void* w, void* y,
    int n_rows, float eps, hipStream_t stream);

void launch(torch::Tensor x, torch::Tensor w, torch::Tensor y, double eps) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "x must be [n_rows, 128]");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == 128, "w must be [128]");
    TORCH_CHECK(y.sizes() == x.sizes(), "y must match x shape");
    rmsnorm_h128_launch(
        x.data_ptr(), w.data_ptr(), y.data_ptr(),
        static_cast<int>(x.size(0)), static_cast<float>(eps),
        c10::hip::getCurrentHIPStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "Launch hip_stock rmsnorm h=128 kernel");
}
