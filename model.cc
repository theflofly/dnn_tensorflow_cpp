#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"

#include "data_set.h"

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;

int main() {
  DataSet data_set("tensorflow/cc/models/", "normalized_car_features.csv");
  Tensor x_data(DataTypeToEnum<float>::v(),
                TensorShape{static_cast<int>(data_set.x().size())/3, 3});
  copy_n(data_set.x().begin(), data_set.x().size(),
         x_data.flat<float>().data());

  Tensor y_data(DataTypeToEnum<float>::v(),
                TensorShape{static_cast<int>(data_set.y().size()), 1});
  copy_n(data_set.y().begin(), data_set.y().size(),
         y_data.flat<float>().data());

  Scope scope = Scope::NewRootScope();

  auto x = Placeholder(scope, DT_FLOAT);
  auto y = Placeholder(scope, DT_FLOAT);

  // weights init
  auto w1 = Variable(scope, {3, 3}, DT_FLOAT);
  auto assign_w1 = Assign(scope, w1, RandomNormal(scope, {3, 3}, DT_FLOAT));

  auto w2 = Variable(scope, {3, 2}, DT_FLOAT);
  auto assign_w2 = Assign(scope, w2, RandomNormal(scope, {3, 2}, DT_FLOAT));

  auto w3 = Variable(scope, {2, 1}, DT_FLOAT);
  auto assign_w3 = Assign(scope, w3, RandomNormal(scope, {2, 1}, DT_FLOAT));

  // bias init
  auto b1 = Variable(scope, {1, 3}, DT_FLOAT);
  auto assign_b1 = Assign(scope, b1, RandomNormal(scope, {1, 3}, DT_FLOAT));

  auto b2 = Variable(scope, {1, 2}, DT_FLOAT);
  auto assign_b2 = Assign(scope, b2, RandomNormal(scope, {1, 2}, DT_FLOAT));

  auto b3 = Variable(scope, {1, 1}, DT_FLOAT);
  auto assign_b3 = Assign(scope, b3, RandomNormal(scope, {1, 1}, DT_FLOAT));

  // layers
  auto layer_1 = Tanh(scope, Tanh(scope, Add(scope, MatMul(scope, x, w1), b1)));
  auto layer_2 = Tanh(scope, Add(scope, MatMul(scope, layer_1, w2), b2));
  auto layer_3 = Tanh(scope, Add(scope, MatMul(scope, layer_2, w3), b3));

  // regularization
  auto regularization = AddN(scope,
                             initializer_list<Input>{L2Loss(scope, w1),
                                                     L2Loss(scope, w2),
                                                     L2Loss(scope, w3)});

  // loss calculation
  auto loss = Add(scope,
                  ReduceMean(scope, Square(scope, Sub(scope, layer_3, y)), {0, 1}),
                  Mul(scope, Cast(scope, 0.01,  DT_FLOAT), regularization));

  // add the gradients operations to the graph
  std::vector<Output> grad_outputs;
  TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w1, w2, w3, b1, b2, b3}, &grad_outputs));

  // update the weights and bias using gradient descent
  auto apply_w1 = ApplyGradientDescent(scope, w1, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[0]});
  auto apply_w2 = ApplyGradientDescent(scope, w2, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[1]});
  auto apply_w3 = ApplyGradientDescent(scope, w3, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[2]});
  auto apply_b1 = ApplyGradientDescent(scope, b1, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[3]});
  auto apply_b2 = ApplyGradientDescent(scope, b2, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[4]});
  auto apply_b3 = ApplyGradientDescent(scope, b3, Cast(scope, 0.01,  DT_FLOAT), {grad_outputs[5]});

  ClientSession session(scope);
  std::vector<Tensor> outputs;
  
  // init the weights and biases by running the assigns nodes once
  TF_CHECK_OK(session.Run({assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3}, nullptr));
  
  // training steps
  for (int i = 0; i < 5000; ++i) {
    TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {loss}, &outputs));
    if (i % 100 == 0) {
      std::cout << "Loss after " << i << " steps " << outputs[0].scalar<float>() << std::endl;
    }
    // nullptr because the output from the run is useless
    TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {apply_w1, apply_w2, apply_w3, apply_b1, apply_b2, apply_b3, layer_3}, nullptr));
  }

  // prediction using the trained neural net
  TF_CHECK_OK(session.Run({{x, {data_set.input(110000.f, Fuel::DIESEL, 7.f)}}}, {layer_3}, &outputs));
  cout << "DNN output: " << *outputs[0].scalar<float>().data() << endl;
  std::cout << "Price predicted " << data_set.output(*outputs[0].scalar<float>().data()) << " euros" << std::endl;

  // saving the model
  //GraphDef graph_def;
  //TF_ASSERT_OK(scope.ToGraphDef(&graph_def));

  return 0;
}