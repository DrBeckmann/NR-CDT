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

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁₁ = render(OrbAndCross(Circle(),Star(1)), width=4)

# ╔═╡ 79449727-86d4-45b7-b4c1-9ac2fcd88c52
J₁₁ = extend_image(I₁₁, (256, 256));

# ╔═╡ 43d55941-e2b7-4228-95d4-0c43858d1089
I₂₂ = render(OrbAndCross(Square(),Star(4)), width=4)

# ╔═╡ f9d33d62-4b48-4878-b504-4d1d00f79c5a
J₂₂ = extend_image(I₂₂, (256, 256));

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
J = [J₁₁, J₂₂]; Label = collect(1:2);

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=90);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.0), 
	scale_y = (0.75, 1.0),
	rotate=(-180.0, 180.0), 
	#shear_x=(-45.0, 45.0),
	#shear_y=(-45.0, 45.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
S = DataTransformations.SaltNoise((4,7), (3/128, 3/128))

# ╔═╡ c8585729-1dc6-437d-807f-f04896f067f1
E = DataTransformations.ElasticNoise(
	amplitude_x=(2.5, 7.5), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TClass = S.(A.(E.(Class)))
# TClass = N.(E.(Class))
# TClass = S.(B.(A.(Class)))

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(850,256,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(64, R)

# ╔═╡ 83ac70b2-b26f-4b30-b4d9-32ac732de5ce
rcdt = RCDT.(TClass);

# ╔═╡ cf0db9d1-1a1f-451d-9e1b-f3bdf02eb413
rqTClass = filter_angles.(rcdt, 128, 128)

# ╔═╡ 439c6bd5-ff76-48d2-aff9-0a4a92136ea7
mqTClass = max_normalization.(rqTClass)

# ╔═╡ d3b97625-b687-48aa-849f-1ec0249dfd02
aqTClass = mean_normalization.(rqTClass)

# ╔═╡ 1a3bffcb-75c8-43bc-9105-07af7c320fd0
# ╠═╡ disabled = true
#=╠═╡
for split in [1,3,9]
	@info "train split:" split/90
	
	Random.seed!(42); accuracy_part_svm(20, Int(class_size/10), 90, 12, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128]
		rqTClass = filter_angles.(rcdt, angle, 128)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 90, 12, rqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 12, mqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 12, aqTClass, Labels)
	end
end
  ╠═╡ =#

# ╔═╡ 35e87506-5cef-4306-83bb-5c967eddbe1d
for split in [1]
	@info "train split:" split/90
	
	Random.seed!(42); accuracy_part_svm(20, split, 90, 2, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128,256]
		rqTClass = filter_angles.(rcdt, angle, 256)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, rqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, mqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, aqTClass, Labels)
	end
end

# ╔═╡ 60ca88f5-f420-45b0-9f1b-01405c0beac7
for split in [3]
	@info "train split:" split/90
	
	Random.seed!(42); accuracy_part_svm(20, split, 90, 2, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128,256]
		rqTClass = filter_angles.(rcdt, angle, 256)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, rqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, mqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, aqTClass, Labels)
	end
end

# ╔═╡ aa4be9ea-f83d-4655-aba8-1a400c124c95
for split in [9]
	@info "train split:" split/90
	
	Random.seed!(42); accuracy_part_svm(20, split, 90, 2, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128,256]
		rqTClass = filter_angles.(rcdt, angle, 256)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, rqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, mqTClass, Labels)
		Random.seed!(42); accuracy_part_svm(20, split, 90, 2, aqTClass, Labels)
	end
end

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═79449727-86d4-45b7-b4c1-9ac2fcd88c52
# ╠═43d55941-e2b7-4228-95d4-0c43858d1089
# ╠═f9d33d62-4b48-4878-b504-4d1d00f79c5a
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
# ╠═c8585729-1dc6-437d-807f-f04896f067f1
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═83ac70b2-b26f-4b30-b4d9-32ac732de5ce
# ╠═cf0db9d1-1a1f-451d-9e1b-f3bdf02eb413
# ╠═439c6bd5-ff76-48d2-aff9-0a4a92136ea7
# ╠═d3b97625-b687-48aa-849f-1ec0249dfd02
# ╠═1a3bffcb-75c8-43bc-9105-07af7c320fd0
# ╠═35e87506-5cef-4306-83bb-5c967eddbe1d
# ╠═60ca88f5-f420-45b0-9f1b-01405c0beac7
# ╠═aa4be9ea-f83d-4655-aba8-1a400c124c95
