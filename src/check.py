import paddle
import paddle.utils as paddle_utils



print(paddle.utils.run_check())
print("PaddlePaddle version:", paddle.__version__)
print("CUDA available:", paddle.is_compiled_with_cuda())

if paddle.is_compiled_with_cuda():
    print("CUDA device count:", paddle.device.cuda.device_count())
    print("CUDA device name:", paddle.device.cuda.get_device_name(0))
else:
    print("Running on CPU.")
