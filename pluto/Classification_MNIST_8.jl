### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 03f9300e-692e-11ef-145c-85ecce1e4c7f
begin
	import Pkg
	Pkg.activate("..")
	using Revise
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT
end

# ╔═╡ 75fa041d-f4a0-4cb0-8af7-b09ff2249874
using MLDatasets, Plots, Random;

# ╔═╡ 0eb4cfdc-c875-4d0f-b46e-531e72fc06c3
trainset = MNIST(:train)

# ╔═╡ 907c64eb-30d8-4cbe-af47-3d4b6d72e3ca
Null = trainset[2].features;

# ╔═╡ 997d8a2c-b11b-4d9e-9812-927a75454479
heatmap(Null)

# ╔═╡ 161664b7-c90c-4ff3-b7d4-1f5ccd536459
heatmap(NormalizedRadonCDT.RadonTransform.radon(Float64.(Null), 40, 1060, 0.0))

# ╔═╡ a71828ea-c868-4a39-abf9-4f0a63b10844
number_mnist_1 = [1,7]

# ╔═╡ 4cb99a9d-f66c-4265-becb-c59aec9daf9f
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 7216e9d9-3773-4f3a-a217-1b57315061a7
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 30909840-1fec-4554-b75c-6331bb483be6
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 01d20c6a-4a99-49da-b5bd-603bf4929e38
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 334f842f-51ba-47c2-a2ed-01cac1caea73
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 63319e89-b389-4fbb-91d0-83e1ed50dd3d
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 1370cb50-9821-41e0-8718-b11e737f3c07
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 5000, 42, 8, 8, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ bee87d59-a137-466b-9d95-759f12c4351e
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 8, 8, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 9de47bc0-cee2-439c-a528-687e234380ed
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 8, 8, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ ea6c606a-809a-408b-a4a7-9db0f3cad8cb
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 8, 8, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 973f67ad-8924-498c-b880-7a46d2d8f36f
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 8, 8, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 5c999783-e3ef-4894-a326-4fb05095ae59
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 8, 8, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 8e2b6bc7-5be6-4584-968d-9679ab823b02
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 8, 8, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 310c18af-d323-40e6-8475-93ee4ebe838b
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 8, 8, 2, 2, 0)

# ╔═╡ d841a04d-4cf9-4946-9a10-ae97c0e9599c
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 8, 8, 2, 2, 0)

# ╔═╡ 300ffc47-68d9-4e7f-bc2b-28ee4ad6206b
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 8, 8, 2, 2, 0)

# ╔═╡ 2a0ab916-7d13-4b75-8b68-0896272ed02a
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 8, 8, 2, 2, 0)

# ╔═╡ 946f9fa3-392a-4fde-809f-7e034c5bf089
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 8, 8, 2, 2, 0)

# ╔═╡ 49c73e62-0bf3-4029-8fa0-e36e70dbddba
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 8, 8, 2, 2, 0)

# ╔═╡ dd0aec18-8630-4c67-9a27-a32c8dd46f6d
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 8, 8, 2, 2, 0)

# ╔═╡ Cell order:
# ╠═03f9300e-692e-11ef-145c-85ecce1e4c7f
# ╠═75fa041d-f4a0-4cb0-8af7-b09ff2249874
# ╠═0eb4cfdc-c875-4d0f-b46e-531e72fc06c3
# ╠═907c64eb-30d8-4cbe-af47-3d4b6d72e3ca
# ╠═997d8a2c-b11b-4d9e-9812-927a75454479
# ╠═161664b7-c90c-4ff3-b7d4-1f5ccd536459
# ╠═a71828ea-c868-4a39-abf9-4f0a63b10844
# ╠═4cb99a9d-f66c-4265-becb-c59aec9daf9f
# ╠═7216e9d9-3773-4f3a-a217-1b57315061a7
# ╠═30909840-1fec-4554-b75c-6331bb483be6
# ╠═01d20c6a-4a99-49da-b5bd-603bf4929e38
# ╠═334f842f-51ba-47c2-a2ed-01cac1caea73
# ╠═63319e89-b389-4fbb-91d0-83e1ed50dd3d
# ╠═1370cb50-9821-41e0-8718-b11e737f3c07
# ╠═bee87d59-a137-466b-9d95-759f12c4351e
# ╠═9de47bc0-cee2-439c-a528-687e234380ed
# ╠═ea6c606a-809a-408b-a4a7-9db0f3cad8cb
# ╠═973f67ad-8924-498c-b880-7a46d2d8f36f
# ╠═5c999783-e3ef-4894-a326-4fb05095ae59
# ╠═8e2b6bc7-5be6-4584-968d-9679ab823b02
# ╠═310c18af-d323-40e6-8475-93ee4ebe838b
# ╠═d841a04d-4cf9-4946-9a10-ae97c0e9599c
# ╠═300ffc47-68d9-4e7f-bc2b-28ee4ad6206b
# ╠═2a0ab916-7d13-4b75-8b68-0896272ed02a
# ╠═946f9fa3-392a-4fde-809f-7e034c5bf089
# ╠═49c73e62-0bf3-4029-8fa0-e36e70dbddba
# ╠═dd0aec18-8630-4c67-9a27-a32c8dd46f6d
