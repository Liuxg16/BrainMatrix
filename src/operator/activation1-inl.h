/*!
 * Copyright (c) 2015 by Contributors
 * \file activation1-inl.h
 * \brief Activation1 operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_ACTIVATION1_INL_H_
#define MXNET_OPERATOR_ACTIVATION1_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation1 {
enum Activation1OpInputs {kData};
enum Activation1OpOutputs {kOut};
enum Activation1OpType {kReLU, kSigmoid, kTanh, kSoftReLU};
}  // activation1

struct Activation1Param : public dmlc::Parameter<Activation1Param> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(Activation1Param) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation1::kReLU)
    .add_enum("sigmoid", activation1::kSigmoid)
    .add_enum("tanh", activation1::kTanh)
    .add_enum("softrelu", activation1::kSoftReLU)
    .describe("Activation1 function to be applied.");
  }
};

/**
 * \brief This is the implementation of activation1 operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp>
class Activation1Op : public Operator {
 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    //printf("0000000-activation1-inh-forward !!! ");
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[activation1::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[activation1::kOut].FlatTo2D<xpu, real_t>(s);
    Assign(out, req[activation1::kOut], F<ForwardOp>(data));
    out += data;
    // Use asynchronize complete notification
    // This is only intended as an example of async ops
    if (s != NULL) s->Wait();

    ctx.async_on_complete();
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    //printf("0000000-activation1-inh-backward !!! ");
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> m_out_grad = out_grad[activation1::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> m_out_data = out_data[activation1::kOut].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> m_in_grad = in_grad[activation1::kData].FlatTo2D<xpu, real_t>(s);
    Assign(m_in_grad, req[activation1::kData], F<BackwardOp>(m_out_data) * m_out_grad);
    // Use asynchronize complete notification
    // This is only intended as an example of async ops
    if (s != NULL) s->Wait();
    ctx.async_on_complete();
  }

  virtual ExecType exec_type() const {
    // Use asynchronize complete notification
    // This is only intended as an example of async ops
    return kAsync;
  }
};  // class Activation1Op

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(Activation1Param type);

#if DMLC_USE_CXX11
class Activation1Prop : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(activation1::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new Activation1Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Activation1";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
#if MXNET_USE_CUDNN == 1
    return {out_grad[activation1::kOut], out_data[activation1::kOut], in_data[activation1::kData]};
#else
    return {out_grad[activation1::kOut], out_data[activation1::kOut]};
#endif  // MXNET_USE_CUDNN
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[activation1::kOut], in_grad[activation1::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[activation1::kData], out_data[activation1::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  Activation1Param param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ACTIVATION1_INL_H_
