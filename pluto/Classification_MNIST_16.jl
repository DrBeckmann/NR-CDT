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
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 7216e9d9-3773-4f3a-a217-1b57315061a7
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 30909840-1fec-4554-b75c-6331bb483be6
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 01d20c6a-4a99-49da-b5bd-603bf4929e38
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 334f842f-51ba-47c2-a2ed-01cac1caea73
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 1315845f-c1f5-487d-ad9f-8b8f3d2d4d25
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 1370cb50-9821-41e0-8718-b11e737f3c07
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 5000, 42, 16, 16, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 25b462af-e62a-4e3d-95de-46506b75c9a3
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 16, 16, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ de3ee80e-e83d-46ab-b9e8-30cffb42cfc0
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 16, 16, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 125dbeaa-a718-4dc3-a81d-f68f8f0ea158
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 16, 16, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 4c60787d-1c55-4bbd-92bd-404ad740ec8a
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 16, 16, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ e5069f84-3a06-4da9-9c7a-64f9f3619ccb
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 16, 16, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 2066467a-d4be-4816-82e9-d2db7eb235b5
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 16, 16, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ d455f30b-ab38-4be9-856f-bbc59aa200a4
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 16, 16, 2, 2, 0)

# ╔═╡ ea193981-2ec1-4804-b346-378b0af8cc8e
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 16, 16, 2, 2, 0)

# ╔═╡ 36bd1c55-5325-41d5-9b79-efaab30a8f07
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 16, 16, 2, 2, 0)

# ╔═╡ 0b3aa15c-d431-429d-8fe6-5e1e6bc96675
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 16, 16, 2, 2, 0)

# ╔═╡ cd7c3bde-e982-40e7-bdc5-40069260b86c
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 16, 16, 2, 2, 0)

# ╔═╡ 27b7cd4d-05e6-4e41-b6b2-63a265630c48
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 16, 16, 2, 2, 0)

# ╔═╡ a7c92de1-0070-4c5e-836f-f5191096adc2
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 16, 16, 2, 2, 0)

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
# ╠═1315845f-c1f5-487d-ad9f-8b8f3d2d4d25
# ╠═1370cb50-9821-41e0-8718-b11e737f3c07
# ╠═25b462af-e62a-4e3d-95de-46506b75c9a3
# ╠═de3ee80e-e83d-46ab-b9e8-30cffb42cfc0
# ╠═125dbeaa-a718-4dc3-a81d-f68f8f0ea158
# ╠═4c60787d-1c55-4bbd-92bd-404ad740ec8a
# ╠═e5069f84-3a06-4da9-9c7a-64f9f3619ccb
# ╠═2066467a-d4be-4816-82e9-d2db7eb235b5
# ╠═d455f30b-ab38-4be9-856f-bbc59aa200a4
# ╠═ea193981-2ec1-4804-b346-378b0af8cc8e
# ╠═36bd1c55-5325-41d5-9b79-efaab30a8f07
# ╠═0b3aa15c-d431-429d-8fe6-5e1e6bc96675
# ╠═cd7c3bde-e982-40e7-bdc5-40069260b86c
# ╠═27b7cd4d-05e6-4e41-b6b2-63a265630c48
# ╠═a7c92de1-0070-4c5e-836f-f5191096adc2
