module NRCDT

# export, using, import statements are usually here; we discuss these below

using Images, Plots, ImageDraw, TestImages, JLD, Random, ImageTransformations, CoordinateTransformations, Rotations, Augmentor, Distributions, Statistics
using ImageReconstruction: radon as Radon
using DataInterpolations
using Interpolations: LinearInterpolation as LinInter
using LIBSVM, LIBLINEAR
using NPZ

####
####
#
#
#
####
####

export gen_circle, gen_square, gen_bar, gen_cross, gen_triangle, gen_star, resize, view_temp, gen_temp
export random_mask, random_image_distortion, gen_dataset, create_data, view_data
export signal_to_pdf, cdt, rradon, rcdt
export prepare_data, shuffle_data, split_data, classify_data_NRCDT

####
####
#
#
#
####
####

include("gen_data_functions.jl")
include("gen_temp_functions.jl")
include("radon_transform.jl")
include("ncrdt_classify.jl")

end;