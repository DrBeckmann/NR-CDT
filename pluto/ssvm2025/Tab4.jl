### A Pluto.jl notebook ###
# v0.20.4

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
	using MLDatasets
end

# ╔═╡ 60c6e670-a8a7-45f4-8f20-ae2a6351a27c
md"""
# SSVM 2025 -- Table 4
This pluto notebook reproduces the numerical experiment
for Table 4 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Max-Normalized Radon Cumulative Distribution
  Transform for Limited Data Classification',
  SSVM 2025.
"""

# ╔═╡ d163b8f7-110f-4a3d-9340-b87a3fa5c888
md"""
## Dataset
Load MNIST 
using `MLDatasets`.
Further information about MNIST can be found in

- L. Deng, 
  '[The MNIST database of handwritten digit images 
  for machine learning research]
  (https://doi.org/10.1109/MSP.2012.2211477)',
  *IEEE Signal Processing Magazine* **29**(6),
  141--142 (2012). 

!!! warning "Download MNIST"
	In order to load MNIST
	using `MLDatasets`,
	MNIST has to be downloaded first!
	For this,
	one can activate the enivironment of the project
	and load the dataset one time explicitly.
	Starting a Julia REPL 
	in the main directory of the project,
	this can be done by
	```
	import Pkg
	Pkg.activate
	using MLDatasets
	MNIST(:train)
	```
	Julia then asks to download the corresponding dataset.
"""

# ╔═╡ 81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
trainset = MNIST(:train)

# ╔═╡ 5b130423-33f7-4b94-a919-9980f5cc7998
md"""
Generate LinMNIST
by selecting a subset of samples
and applying random affine transformations
using the submodule `DataTransformations`.
Further information about LinMNIST can be found in

- M. Beckmann, N. Heilenkötter,
  '[Equivariant neural networks 
  for indirect measurements]
  (https://doi.org/10.1137/23M1582862)',
  *SIAM Journal on Mathematics of Data Science* **6**(3),
  579--601 (2024).
"""

# ╔═╡ 81170d86-6140-41ce-a1e4-24e70c0530ff
MLClass, MLLabel = generate_ml_classes(trainset, [1, 7], 5000);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-45.0, 45.0), 
	shear_x=(-10.0, 10.0),
	shear_y=(-10.0, 10.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20))

# ╔═╡ fb3629dc-1860-4a96-a75e-2b4402f847fe
TMLClass = A.(MLClass)

# ╔═╡ 74f73f93-d83b-4179-b021-a5ec8d7edd5c
md"""
## Linear SVM Classification -- Table 4
Train a linear support vector machine (SVM)
to classify the generated dataset.
The first SVM employes the Euclidean representations,
the second SVM the plain RCDT projections,
and the third SVM the max-normalized RCDTs.
We perform a 10-fold cross-validation,
where one tenth is used for training
and nine tenth for testing.
The experiment is repeated 
for different class sizes
and different numbers of used angles.
For training the linear SVM,
we use the Julia package `LIBLINEAR` from

- R.E. Fan, K.W. Chang, C.J. Hsieh, X.R. Wang, C.J. Lin,
  '[LIBLINEAR: A library for large linear classification]
  (https://www.jmlr.org/papers/volume9/fan08a/fan08a.pdf)', 
  *Journal of Machine Learning Research* **9**, 
  1871--1874 (2008).
"""

# ╔═╡ 9dde6e4f-ca20-4843-90fc-19edddd83f75
for class_size in [10,20,50,250,500,1000,5000]
	@info "class size:" class_size
	
	sTMLClass = gTMLClass[1:class_size];
	append!(sTMLClass, gTMLClass[5001:5000+class_size]);
	sMLLabel = gMLLabel[1:class_size];
	append!(sMLLabel, gMLLabel[5001:5000+class_size]);
	
	accuracy_cross_svm(sTMLClass, sMLLabel)
	
	for angle in [2,4,8,16]
		R = RadonTransform(floor(Int,sqrt(2)*256),angle,0.0);
		RCDT = RadonCDT(floor(Int,sqrt(2)*256), R);
		NRCDT = NormRadonCDT(RCDT);
		mNRCDT = MaxNormRadonCDT(RCDT);
		rqMLClass = RCDT.(sTMLClass);
		mqMLClass = mNRCDT.(sTMLClass);
		@info "number of equispaced angles:" angle
		accuracy_cross_svm(rqMLClass, sMLLabel)
		accuracy_cross_svm(mqMLClass, sMLLabel)	
	end
end

# ╔═╡ Cell order:
# ╟─60c6e670-a8a7-45f4-8f20-ae2a6351a27c
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─d163b8f7-110f-4a3d-9340-b87a3fa5c888
# ╠═81783bfb-d7a2-4c18-a4f8-b634f3bbc59b
# ╟─5b130423-33f7-4b94-a919-9980f5cc7998
# ╠═81170d86-6140-41ce-a1e4-24e70c0530ff
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═fb3629dc-1860-4a96-a75e-2b4402f847fe
# ╟─74f73f93-d83b-4179-b021-a5ec8d7edd5c
# ╠═9dde6e4f-ca20-4843-90fc-19edddd83f75
