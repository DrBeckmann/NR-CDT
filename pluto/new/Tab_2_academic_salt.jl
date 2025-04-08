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
	using Random
	Random.seed!(42)
end

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁ = render(OrbAndCross(Circle(),Star(1)));

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁ = extend_image(I₁, (256, 256))

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂ = render(OrbAndCross(Square(),Star(4)));

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂ = extend_image(I₂, (256, 256))

# ╔═╡ e6c9fb45-2ec7-4925-bc2c-efbed91caa46
I₃ = render(Shield(Triangle()));

# ╔═╡ 25220f99-8cbd-4387-b4fd-bb4a0e6fad96
J₃ = extend_image(I₃, (256, 256))

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
# J = [J₁, J₂, J₄]; Label = [1, 2, 3];
J = [J₁, J₂, J₃]; Label = [1, 5, 12];
# J = [J₂, J₃]; Label = [1, 2];
# J = [J₁, J₂]; Label = [1, 2];
# J = [J₃, J₄]; Label = [1, 2];

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=10);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-5.0, 5.0),
	shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
S = DataTransformations.SaltNoise((5,10), (3/128, 3/128))

# ╔═╡ c8585729-1dc6-437d-807f-f04896f067f1
E = DataTransformations.ElasticNoise(
	amplitude_x=(2.5, 7.5), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TClass = S.(A.(Class))
# TClass = N.(E.(Class))
# TClass = S.(B.(A.(Class)))

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(256,128,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(256, R)

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
NRCDT = NormRadonCDT(RCDT)

# ╔═╡ 81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
mNRCDT = MaxNormRadonCDT(RCDT)

# ╔═╡ 9e056393-fa98-435d-8bf9-8b4d0df99606
aNRCDT = MeanNormRadonCDT(RCDT)

# ╔═╡ 553a0f34-84b4-4997-b00d-c90fdb1ae833
qClass = RCDT.(TClass); mqClass = mNRCDT.(TClass); aqClass = aNRCDT.(TClass);

# ╔═╡ 6f89e4df-7af6-4fdd-bf50-029b07ca82c2
qTemp = RCDT.(J); mqTemp = mNRCDT.(J); aqTemp = aNRCDT.(J);

# ╔═╡ e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
mp = plot_quantiles(mqTemp, Label, mqClass, Labels)

# ╔═╡ 80cdcf56-f43f-4988-9ee6-d1b89c4fb2dd
# ╠═╡ disabled = true
#=╠═╡
savefig(mp, "max_quantiles_elastic.pdf")
  ╠═╡ =#

# ╔═╡ dd239355-c9d4-4556-8134-9ab747f83b59
ap = plot_quantiles(aqTemp, Label, aqClass, Labels)

# ╔═╡ 7e608378-9e3e-4bbb-9a2a-5402582498d1
# ╠═╡ disabled = true
#=╠═╡
savefig(ap, "mean_quantiles_elastic.pdf")
  ╠═╡ =#

# ╔═╡ 42cd7372-3edf-4ddd-99b6-9e70672f74e8
accuracy_k_nearest_neighbour(Array{Float64}.(J), Label, Array{Float64}.(TClass), Labels, "euclidean", ret=1);

# ╔═╡ 8822c353-ed62-4c97-8fb9-f94522b0db1f
accuracy_k_nearest_neighbour(Array{Float64}.(J), Label, Array{Float64}.(TClass), Labels, "inf", ret=1);

# ╔═╡ b0a056d3-d727-4a5d-bcae-264d58f6cae2
for angle in [1,2,4,8,16,32,64,128]
	R = RadonTransform(256,angle,0.0);
	RCDT = RadonCDT(256, R);
	NRCDT = NormRadonCDT(RCDT);
	mNRCDT = MaxNormRadonCDT(RCDT);
	aNRCDT = MeanNormRadonCDT(RCDT);
	qClass = RCDT.(TClass);
	qTemp = RCDT.(J);
	mqClass = mNRCDT.(TClass);
	mqTemp = mNRCDT.(J);
	aqClass = aNRCDT.(TClass);
	aqTemp = aNRCDT.(J);
	@info "number of equispaced angles:" angle
	accuracy_k_nearest_neighbour(qTemp, Label, qClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(qTemp, Label, qClass, Labels, "euclidean", ret=1);
	accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1);
	accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "inf", ret=1);
	accuracy_k_nearest_neighbour(aqTemp, Label, aqClass, Labels, "euclidean", ret=1);
end

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═e6c9fb45-2ec7-4925-bc2c-efbed91caa46
# ╠═25220f99-8cbd-4387-b4fd-bb4a0e6fad96
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
# ╠═c8585729-1dc6-437d-807f-f04896f067f1
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═81e32395-78d9-4a5f-b6f0-ba2d6f01c8ee
# ╠═9e056393-fa98-435d-8bf9-8b4d0df99606
# ╠═553a0f34-84b4-4997-b00d-c90fdb1ae833
# ╠═6f89e4df-7af6-4fdd-bf50-029b07ca82c2
# ╠═e1d7ede5-1b44-4186-9b9e-21e6ddd29dda
# ╠═80cdcf56-f43f-4988-9ee6-d1b89c4fb2dd
# ╠═dd239355-c9d4-4556-8134-9ab747f83b59
# ╠═7e608378-9e3e-4bbb-9a2a-5402582498d1
# ╠═42cd7372-3edf-4ddd-99b6-9e70672f74e8
# ╠═8822c353-ed62-4c97-8fb9-f94522b0db1f
# ╠═b0a056d3-d727-4a5d-bcae-264d58f6cae2
