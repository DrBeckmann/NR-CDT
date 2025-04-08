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
	using Random
	Random.seed!(42)
end

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 98131234-5ab0-4954-bd68-0646241ed22a
typeof(trainset)

# ╔═╡ 81170d86-6140-41ce-a1e4-24e70c0530ff
MLClass, MLLabel = generate_ml_classes(trainset, [1, 7], 500, 0.666);

# ╔═╡ bf8448db-7cb1-42ba-9f1e-03b775b31cb8
MLClass

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
	amplitude_x=(2.5, 7.5), 
	amplitude_y=(2.5, 7.5),
	frequency_x=(0.5, 2.0),
	frequency_y=(0.5, 2.0))

# ╔═╡ 6261bb3f-cbae-462c-9ec1-0a41089904ea
S = DataTransformations.SaltNoise((5,10), (3/128, 3/128))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
Random.seed!(42); TMLClass = S.(A.(MLClass))

# ╔═╡ 8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
R = RadonTransform(128,128,0.0);

# ╔═╡ bbbcd04c-8b4f-4c44-958d-9e4089ada051
RCDT = RadonCDT(128, R);

# ╔═╡ 81fd55d8-24df-4047-b235-20468b2c111c
rcdt = RCDT.(TMLClass);

# ╔═╡ 9dde6e4f-ca20-4843-90fc-19edddd83f75
for split in [1,2] # 10,20,50,250,500,1000,5000
	@info "training split:" split/500
	
	Random.seed!(42); accuracy_part_svm(20, split, 500, 2, TMLClass, MLLabel)
	
	for angle in [2,4,8,16,32,64,128]
		rqMLClass = filter_angles.(rcdt, angle, 128)
		# mqMLClass = mNRCDT.(gTMLClass);
		mqMLClass = max_normalization.(rqMLClass)
		# aqMLClass = aNRCDT.(gTMLClass);
		aqMLClass = mean_normalization.(rqMLClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, rqMLClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, mqMLClass, MLLabel)	
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, aqMLClass, MLLabel)	
	end
end

# ╔═╡ 62d6dcb6-533b-4cf7-8612-f570604a4c75
for split in [5,10] # 10,20,50,250,500,1000,5000
	@info "training split:" split/500
	
	Random.seed!(42); accuracy_part_svm(20, split, 500, 2, TMLClass, MLLabel)
	
	for angle in [2,4,8,16,32,64,128]
		rqMLClass = filter_angles.(rcdt, angle, 128)
		# mqMLClass = mNRCDT.(gTMLClass);
		mqMLClass = max_normalization.(rqMLClass)
		# aqMLClass = aNRCDT.(gTMLClass);
		aqMLClass = mean_normalization.(rqMLClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, rqMLClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, mqMLClass, MLLabel)	
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, aqMLClass, MLLabel)	
	end
end

# ╔═╡ 5f0ee03d-a850-4905-8dc5-030f757c85dc
for split in [25,50] # 10,20,50,250,500,1000,5000
	@info "training split:" split/500
	
	Random.seed!(42); accuracy_part_svm(20, split, 500, 2, TMLClass, MLLabel)
	
	for angle in [2,4,8,16,32,64,128]
		rqMLClass = filter_angles.(rcdt, angle, 128)
		# mqMLClass = mNRCDT.(gTMLClass);
		mqMLClass = max_normalization.(rqMLClass)
		# aqMLClass = aNRCDT.(gTMLClass);
		aqMLClass = mean_normalization.(rqMLClass)
		@info "number of equispaced angles:" angle
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, rqMLClass, MLLabel)
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, mqMLClass, MLLabel)	
		Random.seed!(42); accuracy_part_svm(20, split, 500, 2, aqMLClass, MLLabel)	
	end
end

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╠═98131234-5ab0-4954-bd68-0646241ed22a
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═bf8448db-7cb1-42ba-9f1e-03b775b31cb8
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═0aa6aa3f-3b35-47db-8917-4980c162d6b4
# ╠═6261bb3f-cbae-462c-9ec1-0a41089904ea
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╠═8fb1f5c3-386e-4117-9b87-dedb75c1ae1d
# ╠═bbbcd04c-8b4f-4c44-958d-9e4089ada051
# ╠═81fd55d8-24df-4047-b235-20468b2c111c
# ╠═9dde6e4f-ca20-4843-90fc-19edddd83f75
# ╠═62d6dcb6-533b-4cf7-8612-f570604a4c75
# ╠═5f0ee03d-a850-4905-8dc5-030f757c85dc
