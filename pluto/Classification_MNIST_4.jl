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
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 7216e9d9-3773-4f3a-a217-1b57315061a7
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 30909840-1fec-4554-b75c-6331bb483be6
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 01d20c6a-4a99-49da-b5bd-603bf4929e38
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 334f842f-51ba-47c2-a2ed-01cac1caea73
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 8dccb6bb-ddd3-45b3-8da6-7313eb37e86f
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ 1370cb50-9821-41e0-8718-b11e737f3c07
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 5000, 42, 4, 4, 2, 0, 0)
  ╠═╡ =#

# ╔═╡ e0e18d97-e146-49e9-b218-daf2ceac7cd7
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 4, 4, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 3229a42b-1322-48d4-8b5f-7e192f2f2792
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 4, 4, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 214ccdd3-71a3-4101-a092-4edef9245d75
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 4, 4, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 7437e9bc-f80c-4cf9-93b0-edebda3ac12c
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 4, 4, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 4ac91e4d-86f5-4dc5-a5ab-1a7069f3d3cb
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 4, 4, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ fea80a2e-56de-4dd0-b479-a85b47fc8a96
# ╠═╡ disabled = true
#=╠═╡
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 4, 4, 2, 3.3, 0)
  ╠═╡ =#

# ╔═╡ 1094e986-e97e-40d1-b93c-e4dcc7959254
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 10, 42, 4, 4, 2, 2, 0)

# ╔═╡ 2fd4e68b-8310-4438-ae79-6fe0d2351512
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 20, 42, 4, 4, 2, 2, 0)

# ╔═╡ 9ac6599d-1fbb-4b14-9962-bf7ad81d59fb
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 50, 42, 4, 4, 2, 2, 0)

# ╔═╡ 8906e487-077e-45b7-bfa4-6631363761bf
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 100, 42, 4, 4, 2, 2, 0)

# ╔═╡ 0991cab1-c594-4fd7-8c66-30d2835fc0ea
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 250, 42, 4, 4, 2, 2, 0)

# ╔═╡ f591fba5-1efb-4876-9fa3-6995b0bb20b2
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 500, 42, 4, 4, 2, 2, 0)

# ╔═╡ b15ac9f3-b983-4a34-9d05-e6ac9a39b46d
@time NormalizedRadonCDT.classify_mnist_NRCDT(number_mnist_1, 1000, 42, 4, 4, 2, 2, 0)

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
# ╠═8dccb6bb-ddd3-45b3-8da6-7313eb37e86f
# ╠═1370cb50-9821-41e0-8718-b11e737f3c07
# ╠═e0e18d97-e146-49e9-b218-daf2ceac7cd7
# ╠═3229a42b-1322-48d4-8b5f-7e192f2f2792
# ╠═214ccdd3-71a3-4101-a092-4edef9245d75
# ╠═7437e9bc-f80c-4cf9-93b0-edebda3ac12c
# ╠═4ac91e4d-86f5-4dc5-a5ab-1a7069f3d3cb
# ╠═fea80a2e-56de-4dd0-b479-a85b47fc8a96
# ╠═1094e986-e97e-40d1-b93c-e4dcc7959254
# ╠═2fd4e68b-8310-4438-ae79-6fe0d2351512
# ╠═9ac6599d-1fbb-4b14-9962-bf7ad81d59fb
# ╠═8906e487-077e-45b7-bfa4-6631363761bf
# ╠═0991cab1-c594-4fd7-8c66-30d2835fc0ea
# ╠═f591fba5-1efb-4876-9fa3-6995b0bb20b2
# ╠═b15ac9f3-b983-4a34-9d05-e6ac9a39b46d
