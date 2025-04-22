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

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁₁ = render(OrbAndCross(Circle(),Star(1)))

# ╔═╡ 79449727-86d4-45b7-b4c1-9ac2fcd88c52
J₁₁ = extend_image(I₁₁, (256, 256));

# ╔═╡ af494be1-3291-473a-8160-19de1869dd1d
I₁₂ = render(OrbAndCross(Circle(),Star(4)))

# ╔═╡ 5552472d-b396-4321-a02c-751952b18425
J₁₂ = extend_image(I₁₂, (256, 256));

# ╔═╡ 2aace893-44fe-4756-8ac1-5819ca509596
I₁₃ = render(OrbAndCross(Circle(),Star(8)))

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁₃ = extend_image(I₁₃, (256, 256));

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂₁ = render(OrbAndCross(Square(),Star(1)))

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂₁ = extend_image(I₂₁, (256, 256));

# ╔═╡ 43d55941-e2b7-4228-95d4-0c43858d1089
I₂₂ = render(OrbAndCross(Square(),Star(4)))

# ╔═╡ f9d33d62-4b48-4878-b504-4d1d00f79c5a
J₂₂ = extend_image(I₂₂, (256, 256));

# ╔═╡ 59a3e2b7-4591-455f-885f-35a619329ce0
I₂₃ = render(OrbAndCross(Square(),Star(8)))

# ╔═╡ 767979b7-9e8b-4bfb-909a-5a50daef1c06
J₂₃ = extend_image(I₂₃, (256, 256));

# ╔═╡ 2519d179-3383-4bb3-bb19-97376cae9dbc
I₃₁ = render(OrbAndCross(Triangle(),Star(1)))

# ╔═╡ 83f5a7f3-94ea-42a2-b6f4-23ea07ae2357
J₃₁ = extend_image(I₃₁, (256, 256));

# ╔═╡ f03fd686-9bf6-44b2-839d-29f4c470a26d
I₃₂ = render(OrbAndCross(Triangle(),Star(4)))

# ╔═╡ 39e8dc3d-792d-425e-b4d1-04d617b2a338
J₃₂ = extend_image(I₃₂, (256, 256));

# ╔═╡ e6027f6b-1369-46cc-b9fb-399b7a6d0032
I₃₃ = render(OrbAndCross(Triangle(),Star(8)))

# ╔═╡ 4ad9a1c4-54e9-4e09-ac1e-76cc0d39686e
J₃₃ = extend_image(I₃₃, (256, 256));

# ╔═╡ 52c43eb3-c951-4124-9358-94073007df01
I₄₁ = render(Shield(Circle()))

# ╔═╡ 4305f1da-4d1f-4264-8c90-055a5127b917
J₄₁ = extend_image(I₄₁, (256, 256));

# ╔═╡ feae7daf-7267-4543-9707-286e52b15db7
I₄₂ = render(Shield(Square()))

# ╔═╡ f4dc4243-8223-48c1-bde5-4144072cc94e
J₄₂ = extend_image(I₄₂, (256, 256));

# ╔═╡ 875e9a13-7d49-4669-bdd6-f819f571f2d6
I₄₃ = render(Shield(Triangle()))

# ╔═╡ cad515d1-c4ed-47c2-90f9-b8b88ee30ded
J₄₃ = extend_image(I₄₃, (256, 256));

# ╔═╡ a2ce201f-456c-449b-9d4a-34b02a7579c3
I₄ = render(OrbAndCross(Triangle(),Star(5)));

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
J = [J₁₁, J₁₂, J₁₃, J₂₁, J₂₂, J₂₃, J₃₁, J₃₂, J₃₃, J₄₁, J₄₂, J₄₃]; Label = collect(1:12);

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, class_size=270);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-2.0, 2.0),
	shear_y=(-2.0, 2.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ f3b3dbdd-9797-40ed-aa8a-e589d9be779a
sA = DataTransformations.RandomAffineTransformation(
	scale_x = (1.0, 1.0), 
	scale_y = (1.0, 1.0),
	rotate=(-45.0, 45.0), 
	shear_x=(-5.0, 5.0),
	shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ b26e89d6-a6ee-45d5-a091-acf8c51743d9
N = DataTransformations.MikadoNoise((5,15), (0.125,0.25), (1.5,2.5))

# ╔═╡ 1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
S = DataTransformations.SaltNoise((7,10), (3/128, 3/128))

# ╔═╡ a0358bc3-c54d-4f18-86fc-5578d35a305a
B = DataTransformations.BarNoise((2,6),(2.0, 3.0))

# ╔═╡ c8585729-1dc6-437d-807f-f04896f067f1
E = DataTransformations.ElasticNoise(
	amplitude_x=(2.5, 7.5), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TClass = S.(A.(E.(Class)))
# TClass = N.(E.(Class))
# TClass = S.(B.(A.(Class)))

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(floor(Int,sqrt(2)*128),128,0.0)

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(floor(Int,sqrt(2)*128), R)

# ╔═╡ 83ac70b2-b26f-4b30-b4d9-32ac732de5ce
rcdt = RCDT.(TClass);

# ╔═╡ cf0db9d1-1a1f-451d-9e1b-f3bdf02eb413
rqTClass = filter_angles.(rcdt, 128, 128)

# ╔═╡ 439c6bd5-ff76-48d2-aff9-0a4a92136ea7
mqTClass = max_normalization.(rqTClass)

# ╔═╡ d3b97625-b687-48aa-849f-1ec0249dfd02
aqTClass = mean_normalization.(rqTClass)

# ╔═╡ 565f1403-c48d-4a1a-8b32-107079bc8037
#=╠═╡
hh = heatmap(CM/20, size=(550,500), xticks=(1:12, 1:12), yticks=(1:12, 1:12))
  ╠═╡ =#

# ╔═╡ f4f2aaad-e191-40b5-b064-96123a7e4ae0
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "svm_conv_map_academic_128_128_128_90_2.pdf")
  ╠═╡ =#

# ╔═╡ 35e87506-5cef-4306-83bb-5c967eddbe1d
# ╠═╡ disabled = true
#=╠═╡
for class_size in [10]
	@info "class size:" class_size
	
	# sTClass = gTClass[1:class_size];
	# append!(sTClass, gTClass[91:90+class_size]);
	# sLabels = gLabels[1:class_size];
	# append!(sLabels, gLabels[91:90+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 270, 12, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),angle,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT)
		# rqClass = RCDT.(TClass);
		# mqClass = mNRCDT.(TClass);
		# aqClass = aNRCDT.(TClass);
		rqTClass = filter_angles.(rcdt, angle, 128)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 270, 12, rqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, mqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, aqTClass, Labels)
	end
end
  ╠═╡ =#

# ╔═╡ 1c28418a-25cd-4307-985f-fdaad0a2e9bc
# ╠═╡ disabled = true
#=╠═╡
for class_size in [30]
	@info "class size:" class_size
	
	# sTClass = gTClass[1:class_size];
	# append!(sTClass, gTClass[91:90+class_size]);
	# sLabels = gLabels[1:class_size];
	# append!(sLabels, gLabels[91:90+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 270, 12, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),angle,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT)
		# rqClass = RCDT.(TClass);
		# mqClass = mNRCDT.(TClass);
		# aqClass = aNRCDT.(TClass);
		rqTClass = filter_angles.(rcdt, angle, 128)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 270, 12, rqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, mqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, aqTClass, Labels)
	end
end
  ╠═╡ =#

# ╔═╡ 34c6e19d-c2f7-4a63-a897-950641ad96b0
# ╠═╡ disabled = true
#=╠═╡
for class_size in [90]
	@info "class size:" class_size
	
	# sTClass = gTClass[1:class_size];
	# append!(sTClass, gTClass[91:90+class_size]);
	# sLabels = gLabels[1:class_size];
	# append!(sLabels, gLabels[91:90+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 270, 12, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),angle,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT)
		# rqClass = RCDT.(TClass);
		# mqClass = mNRCDT.(TClass);
		# aqClass = aNRCDT.(TClass);
		rqTClass = filter_angles.(rcdt, angle, 128)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 270, 12, rqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, mqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, aqTClass, Labels)
	end
end
  ╠═╡ =#

# ╔═╡ 3104bccf-f092-4d7d-b03d-be5f593ee71b
# ╠═╡ disabled = true
#=╠═╡
for class_size in [270]
	@info "class size:" class_size
	
	# sTClass = gTClass[1:class_size];
	# append!(sTClass, gTClass[91:90+class_size]);
	# sLabels = gLabels[1:class_size];
	# append!(sLabels, gLabels[91:90+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 270, 12, TClass, Labels)
	
	for angle in [1,2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),angle,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT)
		# rqClass = RCDT.(TClass);
		# mqClass = mNRCDT.(TClass);
		# aqClass = aNRCDT.(TClass);
		rqTClass = filter_angles.(rcdt, angle, 128)
		mqTClass = max_normalization.(rqTClass)
		aqTClass = mean_normalization.(rqTClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 270, 12, rqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, mqTClass, Labels)
		accuracy_part_svm(20, Int(class_size/10), 270, 12, aqTClass, Labels)
	end
end
  ╠═╡ =#

# ╔═╡ f333d1b5-a5b8-47a0-af0e-5263d43c3df3
# ╠═╡ disabled = true
#=╠═╡
CM = accuracy_part_svm(20, 54, 270, 12, rqTClass, Labels, ret=1)
  ╠═╡ =#

# ╔═╡ ee50fec7-305f-4b4d-880e-381b11cfd5c4
#=╠═╡
CM = accuracy_part_svm(20, 54, 270, 12, aqTClass, Labels, ret=1)
  ╠═╡ =#

# ╔═╡ 4f69bbdb-945f-4658-9fc8-b9f5ed7b1c7e
# ╠═╡ disabled = true
#=╠═╡
CM = accuracy_part_svm(20, 54, 270, 12, TClass, Labels, ret=1)
  ╠═╡ =#

# ╔═╡ 52315d01-8a23-4f01-937e-5d186423ffb7
# ╠═╡ disabled = true
#=╠═╡
CM = accuracy_part_svm(20, 135, 270, 12, mqTClass, Labels, ret=1)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═79449727-86d4-45b7-b4c1-9ac2fcd88c52
# ╠═af494be1-3291-473a-8160-19de1869dd1d
# ╠═5552472d-b396-4321-a02c-751952b18425
# ╠═2aace893-44fe-4756-8ac1-5819ca509596
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═43d55941-e2b7-4228-95d4-0c43858d1089
# ╠═f9d33d62-4b48-4878-b504-4d1d00f79c5a
# ╠═59a3e2b7-4591-455f-885f-35a619329ce0
# ╠═767979b7-9e8b-4bfb-909a-5a50daef1c06
# ╠═2519d179-3383-4bb3-bb19-97376cae9dbc
# ╠═83f5a7f3-94ea-42a2-b6f4-23ea07ae2357
# ╠═f03fd686-9bf6-44b2-839d-29f4c470a26d
# ╠═39e8dc3d-792d-425e-b4d1-04d617b2a338
# ╠═e6027f6b-1369-46cc-b9fb-399b7a6d0032
# ╠═4ad9a1c4-54e9-4e09-ac1e-76cc0d39686e
# ╠═52c43eb3-c951-4124-9358-94073007df01
# ╠═4305f1da-4d1f-4264-8c90-055a5127b917
# ╠═feae7daf-7267-4543-9707-286e52b15db7
# ╠═f4dc4243-8223-48c1-bde5-4144072cc94e
# ╠═875e9a13-7d49-4669-bdd6-f819f571f2d6
# ╠═cad515d1-c4ed-47c2-90f9-b8b88ee30ded
# ╠═a2ce201f-456c-449b-9d4a-34b02a7579c3
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═f3b3dbdd-9797-40ed-aa8a-e589d9be779a
# ╠═b26e89d6-a6ee-45d5-a091-acf8c51743d9
# ╠═1f303cbf-8caf-4c85-8f2a-a1460a4c31c3
# ╠═a0358bc3-c54d-4f18-86fc-5578d35a305a
# ╠═c8585729-1dc6-437d-807f-f04896f067f1
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═83ac70b2-b26f-4b30-b4d9-32ac732de5ce
# ╠═cf0db9d1-1a1f-451d-9e1b-f3bdf02eb413
# ╠═439c6bd5-ff76-48d2-aff9-0a4a92136ea7
# ╠═d3b97625-b687-48aa-849f-1ec0249dfd02
# ╠═4f69bbdb-945f-4658-9fc8-b9f5ed7b1c7e
# ╠═f333d1b5-a5b8-47a0-af0e-5263d43c3df3
# ╠═52315d01-8a23-4f01-937e-5d186423ffb7
# ╠═ee50fec7-305f-4b4d-880e-381b11cfd5c4
# ╠═565f1403-c48d-4a1a-8b32-107079bc8037
# ╠═f4f2aaad-e191-40b5-b064-96123a7e4ae0
# ╠═35e87506-5cef-4306-83bb-5c967eddbe1d
# ╠═1c28418a-25cd-4307-985f-fdaad0a2e9bc
# ╠═34c6e19d-c2f7-4a63-a897-950641ad96b0
# ╠═3104bccf-f092-4d7d-b03d-be5f593ee71b
