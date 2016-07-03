/*!
 * Copyright (c) 2015 by Contributors
 * \file activation1.cc
 * \brief activation1 op
 * \author Bing Xu
*/
#include "./activation1-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(Activation1Param param) {

  switch (param.act_type) {
    case activation1::kReLU:
      return new Activation1Op<cpu, mshadow_op::relu, mshadow_op::relu_grad>();
    case activation1::kSigmoid:
      return new Activation1Op<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>();
    case activation1::kTanh:
      return new Activation1Op<cpu, mshadow_op::tanh, mshadow_op::tanh_grad>();
    case activation1::kSoftReLU:
      return new Activation1Op<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>();
    default:
      LOG(FATAL) << "unknown activation1 type";
      return NULL;
  }
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *Activation1Prop::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(Activation1Param);

MXNET_REGISTER_OP_PROPERTY(Activation1, Activation1Prop)
.describe("Apply activation1 function to input."
          "Softmax Activation1 is only available with CUDNN on GPU"
          "and will be computed at each location across channel if input is 4D.")
.add_argument("data", "Symbol", "Input data to activation1 function.")
.add_arguments(Activation1Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet

