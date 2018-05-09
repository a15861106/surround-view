cc_library(
    name = "image_diffuse",
    visibility = ["//visibility:public"],
    srcs = [
        "convolution.h",
        "convolution_loop.h",
        "double_size.h",
        "fill_region.h",
        "half_size.h",
        "image_diffuse.h",
        "kernel.h",
        #"wimage.cc",
        #"wimage.h",
        "macros.h",
    ],
    hdrs = ["image_diffuse.h",],
    deps = [
      "//base",
      "@opencv//:opencv",
    ],
)

cc_binary(
    name = "image_diffuse_test",
    srcs = [
        "image_diffuse_test.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":image_diffuse",
    ],
    linkopts = [
      "-lm",
    ],
)

