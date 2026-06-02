// PyBind11 wrapper for v1_vector_loads/kernel.hip.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

extern "C" void rmsnorm_launch(
    const void* x, const void* w, void* y,
    int n_rows, int hidden, float eps, hipStream_t stream);

void launch(torch::Tensor x, torch::Tensor w, torch::Tensor y, double eps) {
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [n_rows, hidden]");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == x.size(1), "w must be [hidden]");
    TORCH_CHECK(y.sizes() == x.sizes(), "y must match x shape");
    rmsnorm_launch(
        x.data_ptr(), w.data_ptr(), y.data_ptr(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1)),
        static_cast<float>(eps),
        c10::hip::getCurrentHIPStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "v1_vector_loads RMSNorm");
}
