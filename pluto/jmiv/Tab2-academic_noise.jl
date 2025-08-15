### A Pluto.jl notebook ###
# v0.19.47

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
	using NormalizedRadonCDT.Classify
	using Images
	using Plots
end

# ╔═╡ 54002a6c-0cb1-443f-b7f3-7eddd8b4beae
using Random; Random.seed!(42)

# ╔═╡ 420681e5-9a7e-43cb-bae8-b92e8703fa40
using Clustering

# ╔═╡ bdf92971-170c-42ab-a5d9-9fe5ce3fdc6c
md"""
# SSVM 2025 -- Table 2 (left), Figure 4
This pluto notebook reproduces the numerical experiment
for Table 2 (left) and Figure 4 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Max-Normalized Radon Cumulative Distribution
  Transform for Limited Data Classification',
  SSVM 2025.
"""

# ╔═╡ 5b58bf37-7019-4b8d-9dd4-36c04040a393
md"""
## Templates
Generate the three templates
using the submodule `TestImages`.
"""

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁ = render(OrbAndCross(Circle(),Star(1)), width=8);

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁ = extend_image(I₁, (256, 256))

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂ = render(OrbAndCross(Square(),Star(4)), width=8);

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂ = extend_image(I₂, (256, 256))

# ╔═╡ e6c9fb45-2ec7-4925-bc2c-efbed91caa46
I₃ = render(Shield(Triangle()), width=8);

# ╔═╡ 25220f99-8cbd-4387-b4fd-bb4a0e6fad96
J₃ = extend_image(I₃, (256, 256))

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
J = [J₁, J₂, J₃]; Label = [1, 2, 3];
# J = [J₁, J₃]; Label = [1, 3];

# ╔═╡ 0a3ccf51-e303-4f8d-b627-0676df4f561d
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=10);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═╡ disabled = true
#=╠═╡
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-2.0, 2.0),
	shear_y=(-2.0, 2.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))
  ╠═╡ =#

# ╔═╡ 9270a30e-df21-4c2b-8abd-dd4a07b17479
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.25), 
	scale_y = (0.5, 1.25),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 646ac071-c4b4-469f-8f61-6d66bbf2277d
S = DataTransformations.SaltNoise((4,14), (6/128, 6/128), (0,0))

# ╔═╡ f02270be-1d37-425b-ada2-34ead4802cda
B = DataTransformations.BarNoise((3,3), (4,10))

# ╔═╡ dbff202b-d4b1-42bb-b98b-9a46ef329da5
M = DataTransformations.MikadoNoise((5,5), (0.25,1), (2,2))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TClass = S.(A.(Class))

# ╔═╡ 16aaed33-371e-428f-8ba5-fd56d4ee1b76
md"""
## Max-Normalized RCDT -- Figure 4
Setup the max-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 4d57dd04-40c0-44e7-bc6c-f9a30c754d5c
R = RadonTransform(851,64,0.0); RCDT = RadonCDT(64, R); qRCDT = RCDT.(TClass); qtRCDT = RCDT.(J);

# ╔═╡ 0d5bea29-199f-46a8-b0e9-9d102d2789b6
miaqClass = maxminabs_normalization.(qRCDT); miaqTemp = maxminabs_normalization.(qtRCDT); mqClass = max_normalization.(qRCDT); mqTemp = max_normalization.(qtRCDT); iaqClass = minabs_normalization.(qRCDT); iaqTemp = minabs_normalization.(qtRCDT); maqClass = maxabs_normalization.(qRCDT); maqTemp = maxabs_normalization.(qtRCDT); aqClass = mean_normalization.(qRCDT); aqTemp = mean_normalization.(qtRCDT); miqClass = maxmin_normalization.(qRCDT); miqTemp = maxmin_normalization.(qtRCDT); taqTemp = tv_normalization.(qtRCDT); taqClass = tv_normalization.(qRCDT);

# ╔═╡ b03bd8d5-eeaf-4713-85e3-c392bf59dd32
md"""
Plot the computed max-normalized RCDTs.
"""

# ╔═╡ 302adeb9-b396-4888-b2e5-e8e83a15c7e3
plot_quantiles(mqTemp, Label, mqClass, Labels)

# ╔═╡ 6a38a700-942e-4e6e-94b5-3fbb173b9834
plot_quantiles(miqTemp, Label, miqClass, Labels)

# ╔═╡ a29428cb-46a3-4830-a481-894015dd9d27
plot_quantiles(taqTemp[3:3], Label[3:3], taqClass[21:30], Labels[21:30])

# ╔═╡ 536d2775-bb0f-48c9-b51a-e67208803c75
md"""
## Nearest Neighbour Classification -- Table 2
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The max-normalized RCDT is applied
with different numbers of used angles.
"""

# ╔═╡ b0a056d3-d727-4a5d-bcae-264d58f6cae2
# ╠═╡ disabled = true
#=╠═╡
for angle in [2,4,8,16,32,64,128,256]
	R = RadonTransform(851,angle,0.0)
	RCDT = RadonCDT(64, R)
	qRCDT = RCDT.(TClass)
	qtRCDT = RCDT.(J)
	miaqClass = maxminabs_normalization.(qRCDT)
	miaqTemp = maxminabs_normalization.(qtRCDT)
	miqClass = maxmin_normalization.(qRCDT)
	miqTemp = maxmin_normalization.(qtRCDT)
	mqClass = max_normalization.(qRCDT)
	mqTemp = max_normalization.(qtRCDT)
	iaqClass = minabs_normalization.(qRCDT)
	iaqTemp = minabs_normalization.(qtRCDT)
	maqClass = maxabs_normalization.(qRCDT)
	maqTemp = maxabs_normalization.(qtRCDT)
	@info "number of equispaced angles:" angle
	accuracy_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1)
	accuracy_nearest_neighbour(miqTemp, Label, miqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(miqTemp, Label, miqClass, Labels, "euclidean", ret=1)
	accuracy_nearest_neighbour(maqTemp, Label, maqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(maqTemp, Label, maqClass, Labels, "euclidean", ret=1)
	accuracy_nearest_neighbour(iaqTemp, Label, iaqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(iaqTemp, Label, iaqClass, Labels, "euclidean", ret=1)
	accuracy_nearest_neighbour(miaqTemp, Label, miaqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(miaqTemp, Label, miaqClass, Labels, "euclidean", ret=1)
	
end
  ╠═╡ =#

# ╔═╡ 38c64f13-d5cb-4ba9-adce-b75fb719cb1f
# ╠═╡ disabled = true
#=╠═╡
for angle in [16,64,256]
	R = RadonTransform(851,angle,0.0)
	RCDT = RadonCDT(64, R)
	qRCDT = RCDT.(TClass)
	qtRCDT = RCDT.(J)
	miaqClass = maxminabs_normalization.(qRCDT)
	miaqTemp = maxminabs_normalization.(qtRCDT)
	miqClass = maxmin_normalization.(qRCDT)
	miqTemp = maxmin_normalization.(qtRCDT)
	mqClass = max_normalization.(qRCDT)
	mqTemp = max_normalization.(qtRCDT)
	iaqClass = minabs_normalization.(qRCDT)
	iaqTemp = minabs_normalization.(qtRCDT)
	maqClass = maxabs_normalization.(qRCDT)
	maqTemp = maxabs_normalization.(qtRCDT)
	@info "number of equispaced angles:" angle
	for prop in [1,3,5]
		for KK in [1]
			@info "split" prop, "k-NN" KK
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 50, 3, mqClass, Labels, "inf", K=KK, ret=1);
			# jldsave("conf_LinMNIST_$(KK)NN_$(prop)_maxNRCDT_inf.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 50, 3, mqClass, Labels, "euclidean", K=KK, ret=1);
			# jldsave("conf_LinMNIST_$(KK)NN_$(prop)_maxNRCDT_eucl.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 50, 3, miqClass, Labels, "inf", K=KK, ret=1);
			# jldsave("conf_LinMNIST_$(KK)NN_$(prop)_RCDT_inf.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 50, 3, miqClass, Labels, "euclidean", K=KK, ret=1);
			# jldsave("conf_LinMNIST_$(KK)NN_$(prop)_RCDT_eucl.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 50, 3, miaqClass, Labels, "inf", K=KK, ret=1);
			# jldsave("conf_LinMNIST_$(KK)NN_$(prop)_RCDT_inf.jld2"; CC)
			Random.seed!(42); CC = accuracy_k_nearest_part_neighbour(20, prop, 50, 3, miaqClass, Labels, "euclidean", K=KK, ret=1);
			# jldsave("conf_LinMNIST_$(KK)NN_$(prop)_RCDT_eucl.jld2"; CC)
		end
	end
end
  ╠═╡ =#

# ╔═╡ f4f6e184-5dc3-47d9-a7b2-ce40b6c852c1
MiaQClass = reshape(vcat(miaqClass...), :, length(miaqClass));

# ╔═╡ 8000ad18-697e-494e-8124-46c053603b45
RKia = kmeans(MiaQClass, 2, weights=ones(20))

# ╔═╡ e541de09-f559-4d02-be81-e604029499b0
MmQClass = reshape(vcat(mqClass...), :, length(miaqClass));

# ╔═╡ 1df4b3f1-68b8-4ed4-92f0-ad95eaf8c9fd
RKm = kmeans(MmQClass, 2, weights=ones(20))

# ╔═╡ ed92a17c-41d5-4661-aba7-072ae8fdee72
RKia.assignments

# ╔═╡ 0324b4ac-053e-47fc-82d8-f65cf9e96a75
RKm.assignments

# ╔═╡ 301146c6-5152-46f2-8e38-727a78ec1e67
scatter(RKia.centers[:,1], RKia.centers[:,2], group=RKia.assignments)

# ╔═╡ 32a328c6-c62b-45c7-adf1-c09392a61970
scatter(RKm.centers[:,1], RKm.centers[:,2], group=RKm.assignments)

# ╔═╡ Cell order:
# ╟─bdf92971-170c-42ab-a5d9-9fe5ce3fdc6c
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─5b58bf37-7019-4b8d-9dd4-36c04040a393
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═e6c9fb45-2ec7-4925-bc2c-efbed91caa46
# ╠═25220f99-8cbd-4387-b4fd-bb4a0e6fad96
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╟─0a3ccf51-e303-4f8d-b627-0676df4f561d
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═9270a30e-df21-4c2b-8abd-dd4a07b17479
# ╠═646ac071-c4b4-469f-8f61-6d66bbf2277d
# ╠═f02270be-1d37-425b-ada2-34ead4802cda
# ╠═dbff202b-d4b1-42bb-b98b-9a46ef329da5
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─16aaed33-371e-428f-8ba5-fd56d4ee1b76
# ╠═4d57dd04-40c0-44e7-bc6c-f9a30c754d5c
# ╠═0d5bea29-199f-46a8-b0e9-9d102d2789b6
# ╟─b03bd8d5-eeaf-4713-85e3-c392bf59dd32
# ╠═302adeb9-b396-4888-b2e5-e8e83a15c7e3
# ╠═6a38a700-942e-4e6e-94b5-3fbb173b9834
# ╠═a29428cb-46a3-4830-a481-894015dd9d27
# ╟─536d2775-bb0f-48c9-b51a-e67208803c75
# ╠═b0a056d3-d727-4a5d-bcae-264d58f6cae2
# ╠═54002a6c-0cb1-443f-b7f3-7eddd8b4beae
# ╠═38c64f13-d5cb-4ba9-adce-b75fb719cb1f
# ╠═420681e5-9a7e-43cb-bae8-b92e8703fa40
# ╠═f4f6e184-5dc3-47d9-a7b2-ce40b6c852c1
# ╠═8000ad18-697e-494e-8124-46c053603b45
# ╠═e541de09-f559-4d02-be81-e604029499b0
# ╠═1df4b3f1-68b8-4ed4-92f0-ad95eaf8c9fd
# ╠═ed92a17c-41d5-4661-aba7-072ae8fdee72
# ╠═0324b4ac-053e-47fc-82d8-f65cf9e96a75
# ╠═301146c6-5152-46f2-8e38-727a78ec1e67
# ╠═32a328c6-c62b-45c7-adf1-c09392a61970
