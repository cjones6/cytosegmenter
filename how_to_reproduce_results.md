# <center> Reproducing the Experimental Results </center>
# 1. Installation
The code we provide may be run on a CPU only or on a GPU and CPU. We next review and expand on the installation instructions from the Readme. The code has not been tested on Windows operating systems.

We show how to install the dependencies using [Anaconda](https://www.anaconda.com/products/individual). First, create a conda environment. We'll call it `chpt`.
```
conda create -n chpt python=3.7
conda activate chpt
```

Next, install the dependencies for [Chapydette](https://github.com/cjones6/chapydette), our general kernel change-point package. 
```
conda install cython numba numpy scipy scikit-learn
```
If you do not have a GPU and/or want to run the code on only a CPU, you should install the CPU-only versions of the dependencies PyTorch and Faiss via
```
conda install pytorch torchvision cpuonly -c pytorch
conda install faiss-cpu -c pytorch
```
On the other hand, if you have a GPU that is compatible with PyTorch and Faiss you should first determine what version of Cuda you have installed. You can do this by running 
```
nvidia-smi
```
in the terminal. At the top you should see something similar to "CUDA Version: 10.1", with possibly a version different from 10.1. Based on this, install [PyTorch](https://pytorch.org/) and [Faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md), replacing 10.1 below by your Cuda version (see their websites for more details):
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install faiss-gpu cudatoolkit=10.1 -c pytorch
```

In order to compile the Chapydette code you will need to have gcc installed. If you are using Ubuntu, you can install this via
```
sudo apt install build-essential
```
If you are using a Mac, you should install llvm, gcc, and libgcc:
```
conda install llvm gcc libgcc
```

Next, install Chapydette and the remainder of the dependencies. The dependencies basemap, cartopy, and netcdf4 are only used to make the cruise map (Figure 1 in the paper).
```
git clone https://github.com/cjones6/chapydette  
cd chapydette  
python setup.py install
conda install python-dateutil jupyter nb_conda netcdf4 pandas pyarrow seaborn
conda install basemap
conda install -c conda-forge cartopy gsw
conda install matplotlib=3.1.1 --force
```

# 2. Running the scripts
The scripts to reproduce the experimental results are in the `analysis` folder. The scripts assume that the data has been downloaded and is located in a folder `data` in the top level of the directory. That is, relative to the analysis script `1_est_num_cps_phys.py`, the biological data for the cruise DeepDOM, for example, should be located at `../data/DeepDOM_bio.parquet`. The last script assumes that you have downloaded the temperature map found [here](https://psl.noaa.gov/repository/entry/show?entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25vYWEub2lzc3QudjIuaGlnaHJlcy9zc3QuZGF5Lm1lYW4uMjAxNi52Mi5uYw%3D%3D) to the same folder.

Assuming the data is stored in the aforementioned location, one can reproduce the results by sequentially running the scripts in order: 
```
python 1_est_num_cps_phys.py
python 2_generate_features.py
python 3_estimate_cps.py
python 4_create_figures.py
```
Below we provide a brief description of each of these scripts. With an Intel i9-7960X CPU @ 2.80GHz and Titan Xp GPU the code takes around 22 minutes to run. With only the CPU the code takes approximately 13.5 hours to run. 

## 1_est_num_cps_phys.py
The first script estimates the number of change points in each cruise based on the annotations of physical data from all other cruises. We take this leave-one-out approach to demonstrate that even if we don't have annotations for a given cruise, we can still estimate the number of change points in that cruise. We consider two approaches:
1. Using the penalized change-point method of Lebarbier (2005) and tuning the penalty parameter α based on the annotations of other cruises. Concretely, denote by ![\mathcal{L}](https://latex.codecogs.com/svg.latex?\mathcal{L}) the change-point objective with the linear kernel and ![m^{(i)\star}](https://latex.codecogs.com/svg.latex?m^{(i)\star}) the number of annotated change points in auxiliary sequence ![i](https://latex.codecogs.com/svg.latex?i). We tune the penalty ![\alpha](https://latex.codecogs.com/svg.latex?\alpha) of Lebarbier (2005) by minimizing the sum of the absolute difference between the number of true and estimated change points across all cruises ![i=1,\dots,%20S](https://latex.codecogs.com/gif.latex?i%3D1%2C%5Cdots%2CS) except that corresponding to the sequence of interest, which we'll index with ![j](https://latex.codecogs.com/svg.latex?j):  
![\Large \min_{\alpha}\sum_{i\neq j} \left\vert \arg\min_{m}\left[\min_{t_1,\dots, t_m} \mathcal{L}(t_1,\dots, t_m) + \text{pen}_i(\alpha, m) \right]- m^{(i)\star}\right\vert](https://latex.codecogs.com/gif.latex?%5CLarge%20%5Cmin_%7B%5Calpha%7D%5Csum_%7Bi%5Cneq%20j%7D%20%5Cleft%5Cvert%20%5Carg%5Cmin_%7Bm%7D%5Cleft%5B%5Cmin_%7Bt_1%2C%5Cdots%2C%20t_m%7D%20%5Cmathcal%7BL%7D%28t_1%2C%5Cdots%2C%20t_m%29%20&plus;%20%5Ctext%7Bpen%7D_i%28%5Calpha%2C%20m%29%20%5Cright%5D-%20m%5E%7B%28i%29%5Cstar%7D%5Cright%5Cvert)  
where  
![\text{pen}_i(\alpha, m) = \alpha\frac{m+1}{T^{(i)}}\left[2\log\left(\frac{T^{(i)}}{m+1}\right)+5\right]](https://latex.codecogs.com/svg.latex?\text{pen}_i(\alpha,%20m)%20=%20\alpha\frac{m+1}{T^{(i)}}\left[2\log\left(\frac{T^{(i)}}{m+1}\right)+5\right])  
and ![T^{(i)}](https://latex.codecogs.com/gif.latex?T%5E%7B%28i%29%7D) is the length of sequence ![i](https://latex.codecogs.com/svg.latex?i).
2. Using the rule-of-thumb of Harchaoui and Lévy-Leduc (2007) and tuning the parameter ![\nu](https://latex.codecogs.com/svg.latex?\nu) based on the annotations of other cruises. The rule of thumb says to choose the minimum number of change points ![m](https://latex.codecogs.com/svg.latex?m) such that the ratio of successive objective values with ![m+1](https://latex.codecogs.com/svg.latex?m+1) change points and ![m](https://latex.codecogs.com/svg.latex?m) change points exceeds ![1-\nu](https://latex.codecogs.com/svg.latex?1-\nu) for some value ![\nu](https://latex.codecogs.com/svg.latex?\nu). That is, we solve  
![\min_{\nu}\sum_{i\neq j} \left\vert \arg\min_{m}\left[\min_{t_1,\dots, t_m} \mathcal{L}(t_1,\dots, t_m) + \text{pen}_i(\nu, m) \right]- m^{(i)\star}\right\vert](https://latex.codecogs.com/gif.latex?%5Cmin_%7B%5Cnu%7D%5Csum_%7Bi%5Cneq%20j%7D%20%5Cleft%5Cvert%20%5Carg%5Cmin_%7Bm%7D%5Cleft%5B%5Cmin_%7Bt_1%2C%5Cdots%2C%20t_m%7D%20%5Cmathcal%7BL%7D%28t_1%2C%5Cdots%2C%20t_m%29%20&plus;%20%5Ctext%7Bpen%7D_i%28%5Cnu%2C%20m%29%20%5Cright%5D-%20m%5E%7B%28i%29%5Cstar%7D%5Cright%5Cvert)  
where  
![\text{pen}_i({\nu}, m) = 
\begin{cases}
0, \quad \mathcal{L}(t_1,\dots, t_{m+1})/\mathcal{L}(t_1,\dots, t_m) \geq 1-\nu \text{ and } \mathcal{L}(t_1,\dots, t_{m})/\mathcal{L}(t_1,\dots, t_{m-1}) < 1-\nu \\
\infty, \quad \text{else}.
\end{cases}](https://latex.codecogs.com/svg.latex?\begin{align*}\text{pen}_i({\nu},%20m)%20=%20\begin{cases}0,%20\quad%20\mathcal{L}(t_1,\dots,%20t_{m+1})/\mathcal{L}(t_1,\dots,%20t_m)%20\geq%201-\nu%20\text{%20and%20}%20\mathcal{L}(t_1,\dots,%20t_{m})/\mathcal{L}(t_1,\dots,%20t_{m-1})%20%3C%201-\nu%20\\\\\infty,%20\quad%20\text{else}.\end{cases}\end{align*})  

The script saves the optimal ![\alpha](https://latex.codecogs.com/svg.latex?\alpha) and ![\nu](https://latex.codecogs.com/svg.latex?\nu) for each cruise, in addition to the number of estimated change points for each cruise based on the two methods, to the file `../results/estimated_ncp.pickle`.

## 2_generate_features.py
The second script generates features for the biological data and the physical data. For each cruise we do the following. First, we take the log base 10 and then standardize the biological data. We then run k-means on this data with k=128. We store the resultant centroids for use in the Nyström method (Williams and Seeger, 2000). We then apply the Nyström method to every particle measurement vector from the standardized biological data with the aforementioned centroids as the landmarks. This results in features that approximate evaluations of the radial basis function (RBF) kernel. Next, we average the resultant features within each point cloud (3-minute window) to obtain one vector per point cloud. These are the biological features we will use in the main change-point analysis. For the cruise KOK1606 we repeat the above steps with ![k=2^i](https://latex.codecogs.com/svg.latex?k=2^i) for ![i=2,\dots,10](https://latex.codecogs.com/svg.latex?i=2,\dots,10) in order to perform a sensitivity analysis in the next script. We also repeat the above steps on 10 different subsamples of KOK1606 to assess the variability of the estimates. Finally, we generate alternative features where we simply average the data within each point cloud. These alternative features are used to compare our method to a change-in-mean method. To obtain the features for the physical data from each cruise we standardize the physical data from each cruise. 

The script saves the generated biological and physical features, with one file per parameter setting per cruise, in the directory `../features/`.

## 3_estimate_cps.py
The third script applies the kernel change-point estimation method of Harchaoui and Cappé (2007) to the features. For each cruise we do the following. First, we estimate the locations of change points in the physical data. In order to be able to estimate the number of change points, we run the change-point estimation method with the linear kernel for every possible number of change points between 1 and 150. We require a minimum distance of 5 point clouds before change points. We then apply the kernel change-point estimation method on the biological data with the RBF kernel for every possible number of change points between 1 and 150. We again set a minimum distance of 5 point clouds between change points. We set the bandwidth to the median pairwise distance between inputs (a common rule of thumb). After doing this for each cruise, we then repeat the analysis on KOK1606 when using the linear kernel instead of the RBF kernel, varying the bandwidth on a log scale based on the 1st and 99th percentiles of the pairwise distances between inputs, varying the minimum distance between change points between 1 and 50 point clouds, and varying the projection dimension between 4 and 1024. We also repeat it on the 10 subsamples of KOK1606 and on the features from simply averaging the data within each point cloud.

The script saves the estimated physical change points and corresponding objective values for every possible number of change points, with one file per parameter setting per cruise. It similarly saves the estimated biological change points, corresponding objective values, and bandwidth used, with one file per parameter setting per cruise. The results from the main analysis are stored in `../results/`. The results from the sensitivity analysis on KOK1606 are stored in `../results/sensitivity_analysis/` while the results from the variability analysis are stored in `../results/variance_analysis/`. Finally, the results from the comparison with the change-in-mean method are stored in `../results/average_comparison/`.

## 4_create_figures.py
The final script produces all of the figures in the paper. The generated figures are stored in `../plots`. The script also prints values mentioned in the experimental results section of the paper.


# References
- Z. Harchaoui and O. Cappé. Retrospective mutiple change-point estimation with kernels. In *IEEE Workshop on Statistical Signal Processing*, pages 768–772, 2007. 
- Z. Harchaoui and C. Lévy-Leduc. Catching change-points with lasso. In *Advances in Neural Information Processing Systems*, pages 617–624, 2007.
- E. Lebarbier. Detecting multiple change-points in the mean of Gaussian process by model selection. *Signal Processing*, 85(4):717–736, 2005.
- C. K. Williams and M. Seeger. Using the Nyström method to speed up kernel machines. In *Advances in Neural Information Processing Systems*, pages 661–667. MIT press, 2000.