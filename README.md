# NVIDIA AI Workbench: Introduction
This is an [NVIDIA AI Workbench](https://developer.nvidia.com/blog/develop-and-deploy-scalable-generative-ai-models-seamlessly-with-nvidia-ai-workbench/) example Project that provides a short introduction of the cuML library, a Python GPU-accelerated Machine Learning library for building and implementing many common machine learning algorithms and techniques. cuML provides an API built upon Scikit-Learn that will be familiar to ML engineers and developers, so they can use it to easily accelerate their workflows without going into the details of CUDA programming. Users in the [AI Workbench Early Access Program](https://developer.nvidia.com/ai-workbench-early-access) can get up and running with this Project in minutes.

## Project Description
cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other RAPIDS projects. cuML enables data scientists, researchers, and software engineers to run traditional tabular ML tasks on GPUs without going into the details of CUDA programming. In most cases, cuML's Python API matches the API from scikit-learn.

For large datasets, these GPU-based implementations can complete 10-50x faster than their CPU equivalents. 

Included in this project are several easy-to-run tutorial notebooks introducing how to use cuML to work with many common ML algorithms. They are as follows. 

* ```arima_demo.ipynb```: Forecast using ARIMA on time-series data.
  
* ```forest_inference_demo.ipynb```: Save and load an XGBoost model into FIL and infer on new data.
  
* ```kmeans_demo.ipynb```: Predict using k-means, visualize and compare the results with Scikit-learn's k-means.
  
* ```kmeans_mnmg_demo.ipynb```: Predict with Multi-Node Multi-GPU k-means using dask distributed inputs.
  
* ```linear_regression_demo.ipynb```: Demonstrate the use of OLS Linear Regression for prediction.
  
* ```nearest_neighbors_demo.ipynb```: Predict using Nearest Neighbors algorithm.
  
* ```random_forest_demo.ipynb```: Use Random Forest for classification, and demonstrate how to pickle the cuML model.
  
* ```random_forest_mnmg_demo.ipynb```: Solve a classification problem using Multi-Node Multi-GPU Random Forest.

Also included is a benchmarking notebook to evaluate performance between "vanilla" Scikit-Learn and optimized cuML; this may take a while to run, so feel free to push the project to heavier GPU systems to see better performance improvements. Good news: AI Workbench makes this easy!  

* ```cuml_benchmarks.ipynb```: This notebook provides a simple and unified means of benchmarking single GPU cuML algorithms against their skLearn counterparts with the cuml.benchmark package in RAPIDS cuML. This enables quick and simple measurements of performance, validation of correctness, and investigation of upper bounds. Each benchmark returns a Pandas DataFrame with the results. At the end of the notebook, these results are used to draw charts and output to a CSV file. Please refer to the table of contents for algorithms available to be benchmarked with this notebook.

---
**Important Considerations:**
* The notebook titled ```cuml_benchmarks.ipynb``` may take a while to execute since we are running many benchmarks on both regular Scikit-Learn and optimized cuML to evaluate differences in performance. If working on laptop and/or workstation hardware, consider using AI Workbench to push this project to a heavier GPU system to run this notebook. 

---

## System Requirements:
* Operating System: Ubuntu 22.04
* CPU requirements: None, tested with Intel&reg; Xeon&reg; Gold 6240R CPU @ 2.40GHz
* GPU requirements: Any NVIDIA training GPU, tested with NVIDIA A100-40GB
* NVIDIA driver requirements: Latest driver version
* Storage requirements: 40GB

# Quickstart
The notebook(s) in this project were adapted from the RAPIDS cuML Github repository, which can be found [here](https://github.com/rapidsai/cuml/tree/branch-23.10/notebooks).

If you have NVIDIA AI Workbench already installed, you can use this Project in AI Workbench on your choice of machine by:
1. Forking this Project to your own GitHub namespace and copying the clone link

   ```https://github.com/[your_namespace]/<project_name>.git```
   
2. Opening a shell and activating the Context you want to clone into by

   ```
   $ nvwb list contexts
   
   $ nvwb activate <desired_context>
   ```
   
3. Cloning this Project onto your desired machine by running

   ```
   $ nvwb clone project <your_project_url>
   ```
   
4. Opening the Project by

   ```
   $ nvwb list projects
   
   $ nvwb open <project_name>
   ```
   
5. Starting JupyterLab by

   ```
   $ nvwb start jupyterlab
   ```

6. Navigate to the code directory of the project. Then, open the notebooks provided and begin working through them at your own pace. Happy coding!

---
**Tip:** Use ```nvwb help``` to see a full list of commands. 

---

## Tested On
This notebook has been tested with an NVIDIA A100-40gb GPU and an Intel(R) Xeon(R) Gold 6240R CPU (2.40GHz) on the following version of NVIDIA AI Workbench: ```nvwb 0.2.66 (internal; linux; amd64; go1.18.10; Tue Sep 12 18:50:21 UTC 2023)```

## License
This NVIDIA AI Workbench example project is under the [Apache 2.0 License](https://github.com/nv-edwli/rapids-cuml/blob/main/LICENSE.txt)
