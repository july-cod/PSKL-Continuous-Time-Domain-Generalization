# PSKL: Continuous-Time Domain Generalization

This repository provides the implementation of PSKL, a continuous-time domain generalization framework based on Koopman linearization in model parameter space.

## Environment

Clone the repository and download the datasets tracked by Git LFS:

```bash
git clone https://github.com/july-cod/PSKL-Continuous-Time-Domain-Generalization.git
cd PSKL-Continuous-Time-Domain-Generalization
git lfs install
git lfs pull
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate <ENVIRONMENT_NAME>
```

Alternatively, install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

Run an experiment with:

```bash
python main.py --dataset Twitter --cuda 1
```

Replace `Twitter` with the desired dataset name and replace `1` with the GPU index.

Supported dataset names are:

```text
Moons, MNIST, Twitter, YearBook, Cyclone, House
```

For example:

```bash
python main.py --dataset Moons --cuda 0
```

The optional random seed can be specified as follows:

```bash
python main.py --dataset Twitter --cuda 1 --seed 42
```

If CUDA is unavailable, the program automatically uses the CPU.

The dataset configurations are defined in `param.py`.

## Output

The experimental results are saved in:

```text
save/pred_<DATASET_NAME>_<TIMESTAMP>/
```

## License

This project is released under the MIT License.
