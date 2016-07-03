/*!
 * Copyright (c) 2015 by Contributors
 * \file activation1.cu
 * \brief
 * \author Xianggen Liu
*/
#include "./activation1-inl.h"
#include "./mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_activation1-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(Activation1Param param) {
  // SoftReLU not supported by CUDNN yet
  if (param.act_type == activation1::kSoftReLU)
      return new Activation1Op<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>();

#if MXNET_USE_CUDNN == 1
  return new CuDNNActivation1Op(param);
#else
  switch(param.act_type) {
    case activation1::kReLU:
      return new Activation1Op<gpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case activation1::kSigmoid:
      return new Activation1Op<gpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case activation1::kTanh:
      return new Activation1Op<gpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    default:
      LOG(FATAL) << "unknown activation1";
      return NULL;
  }
#endif  // MXNET_USE_CUDNN
}
}  // op
}  // namespace mxnet

