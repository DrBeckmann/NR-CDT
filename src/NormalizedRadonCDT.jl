module NormalizedRadonCDT

using Interpolations: LinearInterpolation
using Statistics

export TestImages
export RadonCDT, RadonTransform
export NormRadonCDT, MaxNormRadonCDT, MeanNormRadonCDT
export RandomAffineTransformation
#export radon_cdt

#using Base.Threads

include("TestImages.jl")
include("RadonTransform.jl")
include("RadonCDT.jl")
include("DataTransformations.jl")
#include("Temp.jl")
#include("Data.jl")
include("Classify.jl")

end
