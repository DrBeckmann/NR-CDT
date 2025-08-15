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
	using Random
	Random.seed!(42)
end

# ╔═╡ bdf92971-170c-42ab-a5d9-9fe5ce3fdc6c
md"""
# SSVM 2025 -- Figure X
This pluto notebook reproduces the numerical visualizations
for Figure X from

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
J = [J₁, J₂, J₃]; Label = [1, 5, 12];

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

# ╔═╡ 9270a30e-df21-4c2b-8abd-dd4a07b17479
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.0), 
	scale_y = (0.75, 1.0),
	rotate=(-180.0, 180.0), 
	#shear_x=(-45.0, 45.0),
	#shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TClass = A.(Class)

# ╔═╡ 16aaed33-371e-428f-8ba5-fd56d4ee1b76
md"""
## Max-Normalized RCDT -- Figure 4
Setup the max-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 4d57dd04-40c0-44e7-bc6c-f9a30c754d5c
R = RadonTransform(501,128,0.0); RCDT = RadonCDT(64, R); qRCDT = RCDT.(TClass); qtRCDT = RCDT.(J);

# ╔═╡ 04dc87af-a19a-4848-b348-2cffbbcf1cf8
nqClass = normalization.(qRCDT);

# ╔═╡ 0d5bea29-199f-46a8-b0e9-9d102d2789b6
miaqClass = maxminabs_normalization.(qRCDT); miaqTemp = maxminabs_normalization.(qtRCDT); mqClass = max_normalization.(qRCDT); mqTemp = max_normalization.(qtRCDT); iaqClass = minabs_normalization.(qRCDT); iaqTemp = minabs_normalization.(qtRCDT); maqClass = maxabs_normalization.(qRCDT); maqTemp = maxabs_normalization.(qtRCDT);  miqClass = maxmin_normalization.(qRCDT); miqTemp = maxmin_normalization.(qtRCDT); taqTemp = tv_normalization.(qtRCDT); taqClass = tv_normalization.(qRCDT); 
# aqClass = mean_normalization.(qRCDT); aqTemp = mean_normalization.(qtRCDT);
# iqClass = min_normalization.(qRCDT); iqTemp = min_normalization.(qtRCDT);
# mtaqTemp = mtv_normalization.(qtRCDT); mtaqClass = mtv_normalization.(qRCDT); 
# meqTemp = mean_normalization.(qtRCDT); meqClass = mean_normalization.(qRCDT); 

# ╔═╡ 55a75e2b-0d51-4209-bd7f-018d14c66a0c
pmax = plot_quantiles(mqTemp, Label, mqClass, Labels)

# ╔═╡ a89ab659-33aa-4512-aba6-c0ad5663141c
savefig(pmax, "max_NRCDT_academic_quantiles_noscale_shear.pdf")

# ╔═╡ 1c31c431-78d4-4bb4-90ef-c3c860fa6cd6
pmaxmin = plot_quantiles(miqTemp, Label, miqClass, Labels)

# ╔═╡ 87109302-7ae2-41cd-9b66-b0f70ae44227
savefig(pmaxmin, "maxmin_NRCDT_academic_quantiles_noscale_shear.pdf")

# ╔═╡ fee8d200-36fe-4ce9-b06b-a98a95489c09
ptv = plot_quantiles(taqTemp, Label, taqClass, Labels)

# ╔═╡ f7d8db9b-84a6-4722-866a-f2d040d82373
savefig(ptv, "tv_NRCDT_academic_quantiles_noscale_shear.pdf")

# ╔═╡ 471b90dc-5777-4976-9b73-f8d6a97cc963
pia = plot_quantiles(iaqTemp, Label, iaqClass, Labels)

# ╔═╡ a64711f5-ffdc-4aa7-8563-d5930505dc5e
savefig(pia, "minabs_NRCDT_academic_quantiles_noscale_shear.pdf")

# ╔═╡ 1cf84256-7fac-4f35-a839-0ff86a584a26
pma = plot_quantiles(maqTemp, Label, maqClass, Labels)

# ╔═╡ 49a1a1d8-2faa-4523-a6d8-211959f213e9
savefig(pma, "maxabs_NRCDT_academic_quantiles_noscale_shear.pdf")

# ╔═╡ 654f5a29-7e86-4265-a2e3-8a117758c1ca
pmia = plot_quantiles(miaqTemp, Label, miaqClass, Labels)

# ╔═╡ 4818db8c-1063-44cf-b95c-4a59185c92ea
savefig(pmia, "maxminabs_NRCDT_academic_quantiles_noscale_shear.pdf")

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
# ╠═╡ disabled = true
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
# ╠═╡ disabled = true
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
# ╠═9270a30e-df21-4c2b-8abd-dd4a07b17479
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─16aaed33-371e-428f-8ba5-fd56d4ee1b76
# ╠═4d57dd04-40c0-44e7-bc6c-f9a30c754d5c
# ╠═04dc87af-a19a-4848-b348-2cffbbcf1cf8
# ╠═0d5bea29-199f-46a8-b0e9-9d102d2789b6
# ╠═55a75e2b-0d51-4209-bd7f-018d14c66a0c
# ╠═a89ab659-33aa-4512-aba6-c0ad5663141c
# ╠═1c31c431-78d4-4bb4-90ef-c3c860fa6cd6
# ╠═87109302-7ae2-41cd-9b66-b0f70ae44227
# ╠═fee8d200-36fe-4ce9-b06b-a98a95489c09
# ╠═f7d8db9b-84a6-4722-866a-f2d040d82373
# ╠═471b90dc-5777-4976-9b73-f8d6a97cc963
# ╠═a64711f5-ffdc-4aa7-8563-d5930505dc5e
# ╠═1cf84256-7fac-4f35-a839-0ff86a584a26
# ╠═49a1a1d8-2faa-4523-a6d8-211959f213e9
# ╠═654f5a29-7e86-4265-a2e3-8a117758c1ca
# ╠═4818db8c-1063-44cf-b95c-4a59185c92ea
# ╟─b03bd8d5-eeaf-4713-85e3-c392bf59dd32
# ╟─536d2775-bb0f-48c9-b51a-e67208803c75
# ╠═b0a056d3-d727-4a5d-bcae-264d58f6cae2
# ╠═fb69725d-8a02-47bf-b8cb-fda0c049feea
