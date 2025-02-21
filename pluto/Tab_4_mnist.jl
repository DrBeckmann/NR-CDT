### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT.DataTransformations
	using Images
	using Plots
	using MLDatasets
end

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 98131234-5ab0-4954-bd68-0646241ed22a
typeof(trainset)

# ╔═╡ 81170d86-6140-41ce-a1e4-24e70c0530ff
MLClass, MLLabel = DataTransformations.samp_mnist(trainset, [1, 7], 50);

# ╔═╡ bf8448db-7cb1-42ba-9f1e-03b775b31cb8
MLClass

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-10.0, 10.0),
	shear_y=(-10.0, 10.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TMLClass = A.(MLClass)

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(256,4,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(256, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ 0d3b5261-8b36-4476-8086-478ba5335517
rClass = RCDT.(TMLClass);

# ╔═╡ 553a0f34-84b4-4997-b00d-c90fdb1ae833
mClass = mNRCDT.(TMLClass);

# ╔═╡ e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
NormalizedRadonCDT.mNRCDT_quantiles([mClass[1], mClass[11]], [1, 7], mClass, MLLabel)

# ╔═╡ 9bfdf99b-6b93-477c-9954-56409f53774a
NormalizedRadonCDT.classify_flatten_svm(TMLClass, MLLabel)

# ╔═╡ 051f04d7-dd1d-4a8e-bd68-32199e4216c0
NormalizedRadonCDT.classify_flatten_svm(rClass, MLLabel)

# ╔═╡ d1208fce-2ef8-4243-99bd-ba6a8f3deda1
NormalizedRadonCDT.classify_flatten_svm(mClass, MLLabel)

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╠═98131234-5ab0-4954-bd68-0646241ed22a
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═bf8448db-7cb1-42ba-9f1e-03b775b31cb8
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═0d3b5261-8b36-4476-8086-478ba5335517
# ╠═553a0f34-84b4-4997-b00d-c90fdb1ae833
# ╠═e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
# ╠═9bfdf99b-6b93-477c-9954-56409f53774a
# ╠═051f04d7-dd1d-4a8e-bd68-32199e4216c0
# ╠═d1208fce-2ef8-4243-99bd-ba6a8f3deda1
