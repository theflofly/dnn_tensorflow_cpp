load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "model",
    srcs = [
        "model.cc",
        "data_set.h",
        "data_set.cc"
    ],
    deps = [
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow"
    ],
    data = ["normalized_car_features.csv"]
)