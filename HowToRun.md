# Run on Docker environment
    1. Build Docker image:
    ```
    docker build -t hicinterpolate .
    ```
    2. Create Docker container:
    ```
    docker run -itd --gpus all -v ${PWD}:${PWD} --name hicinterpolate hicinterpolate:latest
    ```
    3. Enter into Docker container:
    ```
    docker exec -it hicinterpolate bash
    ```

# Run HiCInterpolate 
## Train HiCInterpolate:
    1. Run in distributaed environment:
    ```
    torchrun --standalone --nproc_per_node=2 hicinterpolate.py --distributed --train --test --config config
    ```
    You will find training config file at: `config/config.yaml`

    If you want to create your own dataset, we provided an utiliy script in utils folder named `cool_to_square_matrix.py`
    
    2. After a successful training, you will find all the outputs in the output folder mentioned in config file. We create another folder inside output folder with the same name as config.yaml file itself. Under config folder including best model weights as `hicinterpolate_64.pt` and the last model wrights as `hicinterpolate_64_snapshot.pt`.

## Run our model:
    1. Run the below command:
    ```
    torchrun inference.py --config config
    ```
    You will find inference config file at: `src/inference/config.yaml`

    If you want to create your inference dataset, we provided an utiliy script in utils folder named `inf_cool_to_square_matrix.py`

    2. After a successful training, you will find all the outputs in the output folder mentioned in config file. We create subfolder `inference` and another subfolder inside `inference` folder with the same name as config.yaml file itself. Under config folder, you will find `inferenced.npy` file.

## Run Downstream analysis:
* Run A/B conpartment
```
python dsa.py -ab -i /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data/dmso/chr21_yt.txt -o /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/testing -r 10000 -sc 500 -ec 1000
```
* Run chromatin loops:
```
python dsa.py -l -i /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data/dmso/chr21_yt.txt -o /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/testing -r 10000 -c 21 -g hg38
```
* Run tads:
```
python dsa.py -t -i /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data/dmso/chr21_yt.txt -o /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/data/testing -r 10000
```
* Run 3D structure:
```
docker pull oluwadarelab/hicgnn:latest
docker run --rm -it --name hicgnn_cont -v ${PWD}:/HiC-GNN oluwadarelab/hicgnn
python HiC-GNN_main.py /home/hc0783.unt.ad.unt.edu/workspace/hic_interpolation/analysis_data/raw_data/dmso/chr21_yt.txt
```
For more detainsl for 3D structure predictions, visit: https://github.com/OluwadareLab/HiC-GNN