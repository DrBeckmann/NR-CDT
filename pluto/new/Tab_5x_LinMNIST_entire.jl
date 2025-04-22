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
	using MLDatasets
end

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 98131234-5ab0-4954-bd68-0646241ed22a
typeof(trainset)

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-5.0, 5.0),
	shear_y=(-5.0, 5.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ 0aa6aa3f-3b35-47db-8917-4980c162d6b4
E = DataTransformations.ElasticNoise(
	amplitude_x=(10.0, 20.0), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ 4b9b61f4-df5a-4b9e-aadf-09a7a3f35014
B = DataTransformations.BarNoise((2,6),(2.0, 3.0))

# ╔═╡ 6261bb3f-cbae-462c-9ec1-0a41089904ea
S = DataTransformations.SaltNoise((5,10), (1/128, 7/128))

# ╔═╡ 450ed5c6-d2d2-4399-b3e7-89d821a301ba
N = DataTransformations.MikadoNoise((2,10), (0.125,0.5), (0.1,1))

# ╔═╡ ec24a204-5a8a-4752-8839-6ff7b35756b1
gMLClass, gMLLabel = generate_ml_classes(trainset, [1,2,3,4,5,6,7,8,9,0], 500, 0.666);

# ╔═╡ 44476053-4531-4bc9-9740-44ac54817c44
gTMLClass = A.(gMLClass)

# ╔═╡ 05d26fa9-53de-4383-80b5-d412aa2706af
R = RadonTransform(floor(Int,sqrt(2)*256),128,0.0);

# ╔═╡ afd1db5a-6d63-47ab-9779-0cf48d3e63f9
RCDT = RadonCDT(floor(Int,sqrt(2)*256), R);

# ╔═╡ 53ca36fd-4d14-4152-b67a-c0563bc2ba4f
rcdt = RCDT.(gTMLClass);

# ╔═╡ 8590912e-520e-4393-9539-22e7951fd7d5
for class_size in [10,20] # 10,20,50,250,500,1000,5000
	@info "class size:" class_size
	
	# sTMLClass = gTMLClass[1:class_size];
	# append!(sTMLClass, gTMLClass[1001:1000+class_size]);
	# sMLLabel = gMLLabel[1:class_size];
	# append!(sMLLabel, gMLLabel[1001:1000+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 500, 10, gTMLClass, gMLLabel)
	
	for angle in [2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),128,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT);
		# rqMLClass = RCDT.(gTMLClass);
		rqMLClass = filter_angles.(rcdt, angle, 128)
		# mqMLClass = mNRCDT.(gTMLClass);
		mqMLClass = max_normalization.(rqMLClass)
		# aqMLClass = aNRCDT.(gTMLClass);
		aqMLClass = mean_normalization.(rqMLClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 500, 10, rqMLClass, gMLLabel)
		accuracy_part_svm(20, Int(class_size/10), 500, 10, mqMLClass, gMLLabel)	
		accuracy_part_svm(20, Int(class_size/10), 500, 10, aqMLClass, gMLLabel)	
	end
end

# ╔═╡ 3128e718-f7cb-421e-84d8-8e81ab075f89
for class_size in [50,100] # 10,20,50,250,500,1000,5000
	@info "class size:" class_size
	
	# sTMLClass = gTMLClass[1:class_size];
	# append!(sTMLClass, gTMLClass[1001:1000+class_size]);
	# sMLLabel = gMLLabel[1:class_size];
	# append!(sMLLabel, gMLLabel[1001:1000+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 500, 10, gTMLClass, gMLLabel)
	
	for angle in [2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),128,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT);
		# rqMLClass = RCDT.(gTMLClass);
		rqMLClass = filter_angles.(rcdt, angle, 128)
		# mqMLClass = mNRCDT.(gTMLClass);
		mqMLClass = max_normalization.(rqMLClass)
		# aqMLClass = aNRCDT.(gTMLClass);
		aqMLClass = mean_normalization.(rqMLClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 500, 10, rqMLClass, gMLLabel)
		accuracy_part_svm(20, Int(class_size/10), 500, 10, mqMLClass, gMLLabel)	
		accuracy_part_svm(20, Int(class_size/10), 500, 10, aqMLClass, gMLLabel)	
	end
end

# ╔═╡ f1d5c925-1c2e-4070-a2aa-6ec294b98a9b
for class_size in [250,500] # 10,20,50,250,500,1000,5000
	@info "class size:" class_size
	
	# sTMLClass = gTMLClass[1:class_size];
	# append!(sTMLClass, gTMLClass[1001:1000+class_size]);
	# sMLLabel = gMLLabel[1:class_size];
	# append!(sMLLabel, gMLLabel[1001:1000+class_size]);
	
	accuracy_part_svm(20, Int(class_size/10), 500, 10, gTMLClass, gMLLabel)
	
	for angle in [2,4,8,16,32,64,128]
		# R = RadonTransform(floor(Int,sqrt(2)*128),128,0.0);
		# RCDT = RadonCDT(floor(Int,sqrt(2)*128), R);
		# NRCDT = NormRadonCDT(RCDT);
		# mNRCDT = MaxNormRadonCDT(RCDT);
		# aNRCDT = MeanNormRadonCDT(RCDT);
		# rqMLClass = RCDT.(gTMLClass);
		rqMLClass = filter_angles.(rcdt, angle, 128)
		# mqMLClass = mNRCDT.(gTMLClass);
		mqMLClass = max_normalization.(rqMLClass)
		# aqMLClass = aNRCDT.(gTMLClass);
		aqMLClass = mean_normalization.(rqMLClass)
		@info "number of equispaced angles:" angle
		accuracy_part_svm(20, Int(class_size/10), 500, 10, rqMLClass, gMLLabel)
		accuracy_part_svm(20, Int(class_size/10), 500, 10, mqMLClass, gMLLabel)	
		accuracy_part_svm(20, Int(class_size/10), 500, 10, aqMLClass, gMLLabel)	
	end
end

# ╔═╡ a387e926-841a-4d76-a423-4ac8453f6949
rqMLClass = filter_angles.(rcdt, 128, 128)

# ╔═╡ aba521c2-bbfb-4336-9908-71c5a73afa57
mqMLClass = max_normalization.(rqMLClass)

# ╔═╡ 1d1dc5ec-143b-41f8-90f3-4c3975315a8a
aqMLClass = mean_normalization.(rqMLClass)

# ╔═╡ 68657bd4-aaa2-417e-8b85-16d4c2fb240c
hh = heatmap(CC/20, size=(550,500), xticks=(1:10, 0:9), yticks=(1:10, 0:9))

# ╔═╡ a414f4fb-3ad2-427a-9d37-528ef8cb49ed
# ╠═╡ disabled = true
#=╠═╡
savefig(hh, "conf_map_mnist_11NN_1_max_eucl.pdf")
  ╠═╡ =#

# ╔═╡ 50e6057d-0043-4743-9db7-2b26d6dab5a3
CC = accuracy_k_nearest_part_neighbour(20, 50, 500, 10, mqMLClass, gMLLabel, "euclidean", K=11, ret=1);

# ╔═╡ d24c7075-b34c-4768-9078-b80a844ca252
# ╠═╡ disabled = true
#=╠═╡
CC = accuracy_part_svm(20, 50, 500, 10, mqMLClass, gMLLabel, ret=1)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╠═98131234-5ab0-4954-bd68-0646241ed22a
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═0aa6aa3f-3b35-47db-8917-4980c162d6b4
# ╠═4b9b61f4-df5a-4b9e-aadf-09a7a3f35014
# ╠═6261bb3f-cbae-462c-9ec1-0a41089904ea
# ╠═450ed5c6-d2d2-4399-b3e7-89d821a301ba
# ╠═ec24a204-5a8a-4752-8839-6ff7b35756b1
# ╠═44476053-4531-4bc9-9740-44ac54817c44
# ╠═05d26fa9-53de-4383-80b5-d412aa2706af
# ╠═afd1db5a-6d63-47ab-9779-0cf48d3e63f9
# ╠═53ca36fd-4d14-4152-b67a-c0563bc2ba4f
# ╠═8590912e-520e-4393-9539-22e7951fd7d5
# ╠═3128e718-f7cb-421e-84d8-8e81ab075f89
# ╠═f1d5c925-1c2e-4070-a2aa-6ec294b98a9b
# ╠═a387e926-841a-4d76-a423-4ac8453f6949
# ╠═aba521c2-bbfb-4336-9908-71c5a73afa57
# ╠═1d1dc5ec-143b-41f8-90f3-4c3975315a8a
# ╠═d24c7075-b34c-4768-9078-b80a844ca252
# ╠═50e6057d-0043-4743-9db7-2b26d6dab5a3
# ╠═68657bd4-aaa2-417e-8b85-16d4c2fb240c
# ╠═a414f4fb-3ad2-427a-9d37-528ef8cb49ed
