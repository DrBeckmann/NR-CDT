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
end

# ╔═╡ 92cfdfdf-cfac-40bd-9bf5-bc44b3942e98
md"""
# SSVM 2025 -- Table 3
This pluto notebook reproduces the numerical experiment
for Table 3 from

- Matthias Beckmann, Robert Beinert, Jonas Bresch, 
  'Max-Normalized Radon Cumulative Distribution
  Transform for Limited Data Classification',
  SSVM 2025.
"""

# ╔═╡ 5d298c1f-c203-4f31-b904-0af64d3bd7da
md"""
## Templates
Generate the two templates
using the submodule `TestImages`.
"""

# ╔═╡ c9a1f57f-1874-40e4-b47f-d66f7dd4a064
I₁ = render(OrbAndCross(Circle(),Star(1)));

# ╔═╡ 237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
J₁ = extend_image(I₁, (256, 256))

# ╔═╡ 8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
I₂ = render(OrbAndCross(Square(),Star(4)));

# ╔═╡ 29d43338-99a6-42ce-9cf6-eee91d3905b8
J₂ = extend_image(I₂, (256, 256))

# ╔═╡ c1959bcd-b5ce-40ae-9c41-142fca3027b6
J = [J₁, J₂]; Label = [1, 2];

# ╔═╡ 19430770-d572-4e16-9191-70ae876be076
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ 14864b75-d2e6-476a-bf63-5ffffa95a61d
Class, Labels = generate_academic_classes(J, Label, class_size=270);

# ╔═╡ 773832af-9099-4dcf-bd1b-c82baaa83424
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.75, 1.25), 
	scale_y = (0.75, 1.25),
	rotate=(-90.0, 90.0), 
	shear_x=(-2.0, 2.0),
	shear_y=(-2.0, 2.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20)
	)

# ╔═╡ c974aee4-ba38-4103-ab3d-b9d1a29066e9
TClass = A.(Class)

# ╔═╡ 8e54cefd-8133-41f9-ae1e-69968bd1c9e5
md"""
## Linear SVM Classification -- Table 3
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

# ╔═╡ 59720153-0f97-4144-befa-0cff41aee075
for class_size in [10,30,90,270]
	@info "class size:" class_size
	
	sTClass = TClass[1:class_size];
	append!(sTClass, TClass[271:270+class_size]);
	sLabels = Labels[1:class_size];
	append!(sLabels, Labels[271:270+class_size]);
	
	accuracy_cross_svm(sTClass, sLabels)
	
	for angle in [2,4,8]
		R = RadonTransform(floor(Int,sqrt(2)*256),angle,0.0)
		RCDT = RadonCDT(floor(Int,sqrt(2)*256), R)
		mNRCDT = MaxNormRadonCDT(RCDT)
		rqClass = RCDT.(sTClass)
		mqClass = mNRCDT.(sTClass)
		@info "number of equispaced angles:" angle
		accuracy_cross_svm(rqClass, sLabels)
		accuracy_cross_svm(mqClass, sLabels)	
	end
end

# ╔═╡ Cell order:
# ╟─92cfdfdf-cfac-40bd-9bf5-bc44b3942e98
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─5d298c1f-c203-4f31-b904-0af64d3bd7da
# ╠═c9a1f57f-1874-40e4-b47f-d66f7dd4a064
# ╠═237a6d1c-6d30-40a0-8eb6-aa0ae913b6d2
# ╠═8ab0ffae-2f4c-4b8b-b201-7f86d9ef25ac
# ╠═29d43338-99a6-42ce-9cf6-eee91d3905b8
# ╠═c1959bcd-b5ce-40ae-9c41-142fca3027b6
# ╟─19430770-d572-4e16-9191-70ae876be076
# ╠═14864b75-d2e6-476a-bf63-5ffffa95a61d
# ╠═773832af-9099-4dcf-bd1b-c82baaa83424
# ╠═c974aee4-ba38-4103-ab3d-b9d1a29066e9
# ╟─8e54cefd-8133-41f9-ae1e-69968bd1c9e5
# ╠═59720153-0f97-4144-befa-0cff41aee075
