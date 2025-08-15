### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("../..")
	using Revise
	using NormalizedRadonCDT
	using NormalizedRadonCDT.TestImages
	using NormalizedRadonCDT.DataTransformations
	using NormalizedRadonCDT.Classify
	using Images
	using Plots
end

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

# ╔═╡ 5838d8cc-4bcd-47de-8822-bb48e1f82bcc
I₁₂ = render(OrbAndCross(Circle(),Star(4)), width=8);

# ╔═╡ 99ca6592-5fa0-46ce-a559-4b246f99b910
I₁₃ = render(OrbAndCross(Circle(),Star(8)), width=8);

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁₁ = extend_image(I₁, (256, 256))

# ╔═╡ 3a78a637-43a4-4ccf-9482-4419551e1b5e
J₁₂ = extend_image(I₁₂, (256, 256))

# ╔═╡ debd6489-2cbb-4bd2-a1a2-44fc09c0e5b3
J₁₃ = extend_image(I₁₃, (256, 256))

# ╔═╡ 048f899f-a48f-492d-a5db-20fcc14c0a03
I₂₁ = render(OrbAndCross(Square(),Star(1)), width=8);

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂₂ = render(OrbAndCross(Square(),Star(4)), width=8);

# ╔═╡ 43a2217b-0db3-4aa9-b980-b30a1480019f
I₂₃ = render(OrbAndCross(Square(),Star(8)), width=8);

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂₁ = extend_image(I₂₁, (256, 256))

# ╔═╡ 308d1cb3-2224-4195-8ce0-685d10afb89f
J₂₂ = extend_image(I₂₂, (256, 256))

# ╔═╡ 4cd55a68-4498-4a9a-8f84-0e3d42bc8baf
J₂₃ = extend_image(I₂₃, (256, 256))

# ╔═╡ 2dd7f616-86e5-4560-a3e3-7c33718589a4
I₃₁ = render(Shield(Circle()), width=8);

# ╔═╡ e6c9fb45-2ec7-4925-bc2c-efbed91caa46
I₃₂ = render(Shield(Square()), width=8);

# ╔═╡ a117a7dd-ea68-4787-b6f4-117a30548854
I₃₃ = render(Shield(Triangle()), width=8);

# ╔═╡ 25220f99-8cbd-4387-b4fd-bb4a0e6fad96
J₃₁ = extend_image(I₃₁, (256, 256))

# ╔═╡ 3d28cb13-4b76-4976-8040-d68d6cee5122
J₃₂ = extend_image(I₃₂, (256, 256))

# ╔═╡ 7eb37022-09b9-42b2-9189-bda7b1837aa8
J₃₃ = extend_image(I₃₃, (256, 256))

# ╔═╡ 66f73d64-ae47-448d-9688-5541b7c3b391
I₄₁ = render(OrbAndCross(Triangle(),Star(1)), width=8);

# ╔═╡ a1b31426-9153-48b2-b091-b36aa59e147d
I₄₂ = render(OrbAndCross(Triangle(),Star(4)), width=8);

# ╔═╡ 730007ec-4fdd-4532-8675-b59396e6c300
I₄₃ = render(OrbAndCross(Triangle(),Star(8)), width=8);

# ╔═╡ c232bd2d-61f0-4147-8a50-13157d3b067f
J₄₁ = extend_image(I₄₁, (256, 256))

# ╔═╡ a15af140-9315-4a94-9a3b-0a27cc4b1c2e
J₄₂ = extend_image(I₄₂, (256, 256))

# ╔═╡ a01e2f45-ea6b-4b98-b478-f2e401ba8df5
J₄₃ = extend_image(I₄₃, (256, 256))

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
# J = [J₁, J₂, J₃]; Label = [1, 2, 3];
# J = [J₁, J₂, J₄]; Label = [1, 2, 3];
J = [J₁₁,J₁₂,J₁₃,J₂₁,J₂₂,J₂₃,J₃₁,J₃₂,J₃₃,J₄₁,J₄₂,J₄₃]; Label = [1,2,3,4,5,6,7,8,9,10,11,12];

# ╔═╡ 0a3ccf51-e303-4f8d-b627-0676df4f561d
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=50);

# ╔═╡ 646ac071-c4b4-469f-8f61-6d66bbf2277d
S = DataTransformations.SaltNoise((4,14), (3/128, 3/128), (0,0))

# ╔═╡ f02270be-1d37-425b-ada2-34ead4802cda
B = DataTransformations.BarNoise((3,3), (4,10))

# ╔═╡ dbff202b-d4b1-42bb-b98b-9a46ef329da5
M = DataTransformations.MikadoNoise((3,3), (0.25,1), (2,2))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
#=╠═╡
TClass = A.(Class)
  ╠═╡ =#

# ╔═╡ 16aaed33-371e-428f-8ba5-fd56d4ee1b76
md"""
## Max-Normalized RCDT -- Figure 4
Setup the max-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 4d57dd04-40c0-44e7-bc6c-f9a30c754d5c
#=╠═╡
R = RadonTransform(851,128,0.0); RCDT = RadonCDT(64, R); qRCDT = RCDT.(TClass); qtRCDT = RCDT.(J);
  ╠═╡ =#

# ╔═╡ 04dc87af-a19a-4848-b348-2cffbbcf1cf8
#=╠═╡
nqClass = normalization.(qRCDT);
  ╠═╡ =#

# ╔═╡ 0d5bea29-199f-46a8-b0e9-9d102d2789b6
# ╠═╡ disabled = true
#=╠═╡
miaqClass = maxminabs_normalization.(qRCDT); miaqTemp = maxminabs_normalization.(qtRCDT); mqClass = max_normalization.(qRCDT); mqTemp = max_normalization.(qtRCDT); iaqClass = minabs_normalization.(qRCDT); iaqTemp = minabs_normalization.(qtRCDT); maqClass = maxabs_normalization.(qRCDT); maqTemp = maxabs_normalization.(qtRCDT); aqClass = mean_normalization.(qRCDT); aqTemp = mean_normalization.(qtRCDT); miqClass = maxmin_normalization.(qRCDT); miqTemp = maxmin_normalization.(qtRCDT); iqClass = min_normalization.(qRCDT); iqTemp = min_normalization.(qtRCDT); taqTemp = tv_normalization.(qtRCDT); taqClass = tv_normalization.(qRCDT); mtaqTemp = mtv_normalization.(qtRCDT); mtaqClass = mtv_normalization.(qRCDT); 
# meqTemp = mean_normalization.(qtRCDT); meqClass = mean_normalization.(qRCDT); 
  ╠═╡ =#

# ╔═╡ 55a75e2b-0d51-4209-bd7f-018d14c66a0c
# ╠═╡ disabled = true
#=╠═╡
plot_quantiles(mqTemp, Label, mqClass, Labels)
  ╠═╡ =#

# ╔═╡ 1c31c431-78d4-4bb4-90ef-c3c860fa6cd6
# ╠═╡ disabled = true
#=╠═╡
plot_quantiles(miqTemp, Label, miqClass, Labels)
  ╠═╡ =#

# ╔═╡ fee8d200-36fe-4ce9-b06b-a98a95489c09
# ╠═╡ disabled = true
#=╠═╡
plot_quantiles(taqTemp, Label, taqClass, Labels)
  ╠═╡ =#

# ╔═╡ b03bd8d5-eeaf-4713-85e3-c392bf59dd32
md"""
Plot the computed max-normalized RCDTs.
"""

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
#=╠═╡
for angle in [2,4,8,16,]
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
	tvqClass = tv_normalization.(qRCDT)
	tvqTemp = tv_normalization.(qtRCDT)
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
	accuracy_nearest_neighbour(tvqTemp, Label, tvqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(tvqTemp, Label, tvqClass, Labels, "euclidean", ret=1)
end
  ╠═╡ =#

# ╔═╡ fb69725d-8a02-47bf-b8cb-fda0c049feea
#=╠═╡
for angle in [32,64,128,256]
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
	tvqClass = tv_normalization.(qRCDT)
	tvqTemp = tv_normalization.(qtRCDT)
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
	accuracy_nearest_neighbour(tvqTemp, Label, tvqClass, Labels, "inf", ret=1)
	accuracy_nearest_neighbour(tvqTemp, Label, tvqClass, Labels, "euclidean", ret=1)
end
  ╠═╡ =#

# ╔═╡ f4f6e184-5dc3-47d9-a7b2-ce40b6c852c1
#=╠═╡
MiaQClass = reshape(vcat(taqClass...), :, length(miaqClass));
  ╠═╡ =#

# ╔═╡ 8000ad18-697e-494e-8124-46c053603b45
#=╠═╡
RKia = kmeans(MiaQClass, 3, weights=ones(30))
  ╠═╡ =#

# ╔═╡ e541de09-f559-4d02-be81-e604029499b0
#=╠═╡
MmQClass = reshape(vcat(mqClass...), :, length(miaqClass));
  ╠═╡ =#

# ╔═╡ 1df4b3f1-68b8-4ed4-92f0-ad95eaf8c9fd
#=╠═╡
RKm = kmeans(MmQClass, 3, weights=ones(30))
  ╠═╡ =#

# ╔═╡ ed92a17c-41d5-4661-aba7-072ae8fdee72
#=╠═╡
RKia.assignments
  ╠═╡ =#

# ╔═╡ 0324b4ac-053e-47fc-82d8-f65cf9e96a75
#=╠═╡
RKm.assignments
  ╠═╡ =#

# ╔═╡ 301146c6-5152-46f2-8e38-727a78ec1e67
#=╠═╡
scatter(RKia.centers[:,1], RKia.centers[:,2], group=RKia.assignments)
  ╠═╡ =#

# ╔═╡ 32a328c6-c62b-45c7-adf1-c09392a61970
#=╠═╡
scatter(RKm.centers[:,1], RKm.centers[:,2], group=RKm.assignments)
  ╠═╡ =#

# ╔═╡ 9270a30e-df21-4c2b-8abd-dd4a07b17479
#=╠═╡
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.25), 
	scale_y = (0.5, 1.25),
	rotate=(-180.0, 180.0), 
	shear_x=(-45.0, 45.0),
	shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))
  ╠═╡ =#

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

# ╔═╡ Cell order:
# ╟─bdf92971-170c-42ab-a5d9-9fe5ce3fdc6c
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─5b58bf37-7019-4b8d-9dd4-36c04040a393
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═5838d8cc-4bcd-47de-8822-bb48e1f82bcc
# ╠═99ca6592-5fa0-46ce-a559-4b246f99b910
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═3a78a637-43a4-4ccf-9482-4419551e1b5e
# ╠═debd6489-2cbb-4bd2-a1a2-44fc09c0e5b3
# ╠═048f899f-a48f-492d-a5db-20fcc14c0a03
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═43a2217b-0db3-4aa9-b980-b30a1480019f
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═308d1cb3-2224-4195-8ce0-685d10afb89f
# ╠═4cd55a68-4498-4a9a-8f84-0e3d42bc8baf
# ╠═2dd7f616-86e5-4560-a3e3-7c33718589a4
# ╠═e6c9fb45-2ec7-4925-bc2c-efbed91caa46
# ╠═a117a7dd-ea68-4787-b6f4-117a30548854
# ╠═25220f99-8cbd-4387-b4fd-bb4a0e6fad96
# ╠═3d28cb13-4b76-4976-8040-d68d6cee5122
# ╠═7eb37022-09b9-42b2-9189-bda7b1837aa8
# ╠═66f73d64-ae47-448d-9688-5541b7c3b391
# ╠═a1b31426-9153-48b2-b091-b36aa59e147d
# ╠═730007ec-4fdd-4532-8675-b59396e6c300
# ╠═c232bd2d-61f0-4147-8a50-13157d3b067f
# ╠═a15af140-9315-4a94-9a3b-0a27cc4b1c2e
# ╠═a01e2f45-ea6b-4b98-b478-f2e401ba8df5
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
# ╠═04dc87af-a19a-4848-b348-2cffbbcf1cf8
# ╠═0d5bea29-199f-46a8-b0e9-9d102d2789b6
# ╠═55a75e2b-0d51-4209-bd7f-018d14c66a0c
# ╠═1c31c431-78d4-4bb4-90ef-c3c860fa6cd6
# ╠═fee8d200-36fe-4ce9-b06b-a98a95489c09
# ╟─b03bd8d5-eeaf-4713-85e3-c392bf59dd32
# ╟─536d2775-bb0f-48c9-b51a-e67208803c75
# ╠═b0a056d3-d727-4a5d-bcae-264d58f6cae2
# ╠═fb69725d-8a02-47bf-b8cb-fda0c049feea
# ╠═420681e5-9a7e-43cb-bae8-b92e8703fa40
# ╠═f4f6e184-5dc3-47d9-a7b2-ce40b6c852c1
# ╠═8000ad18-697e-494e-8124-46c053603b45
# ╠═e541de09-f559-4d02-be81-e604029499b0
# ╠═1df4b3f1-68b8-4ed4-92f0-ad95eaf8c9fd
# ╠═ed92a17c-41d5-4661-aba7-072ae8fdee72
# ╠═0324b4ac-053e-47fc-82d8-f65cf9e96a75
# ╠═301146c6-5152-46f2-8e38-727a78ec1e67
# ╠═32a328c6-c62b-45c7-adf1-c09392a61970
