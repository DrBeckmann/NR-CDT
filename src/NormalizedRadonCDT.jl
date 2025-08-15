module NormalizedRadonCDT

using Interpolations: LinearInterpolation
using Statistics

export TestImages
export RadonCDT, RadonTransform, Backprojection, FilterBackprojection
export NormRadonCDT, MaxNormRadonCDT, MeanNormRadonCDT, MinNormRadonCDT, MaxMinAbsNormRadonCDT, MaxMinNormRadonCDT
export normalization, filter_angles, max_normalization, min_normalization, mean_normalization, maxmin_normalization, maxminabs_normalization, minabs_normalization, tv_normalization, mtv_normalization, median_normalization, maxabs_normalization, mink, maxk, absm
export DataTransformations
export Classify

export RadonCDT3d, RadonTransform3d
export NormRadonCDT3d, MaxNormRadonCDT3d
export normalization3d, max_normalization3d, mean_normalization3d


include("TestImages.jl")
include("RadonTransform.jl")
include("RadonCDT.jl")
include("DataTransformations.jl")
include("Classify.jl")

include("RadonTransform3d.jl")
include("RadonCDT3d.jl")

end
