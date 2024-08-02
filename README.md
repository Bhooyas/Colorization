# Image Colorization

![Colorization](https://socialify.git.ci/Bhooyas/Colorization/image?font=KoHo&language=1&name=1&owner=1&pattern=Circuit%20Board&stargazers=1&theme=Auto)

Pytorch implementation of unet architecture for image colorization.

## Usage

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/Colorization.git
```

The next step would be to install the requirements: -
```
cd Colorization
pip install -r requirements.txt
```

The next step will be to get the data. You can run the follwing shell script to get the data.

For bash: -
```
./get_dataset.sh
```

For powershell: -
```
.\get_dataset.ps1
```

The configuration for the model can be found and edited in the `config.py`.

The next step is to create a subset and preprocess the data. We do this using the following command: -
```
python create_subset.py
```

The next step is to train the model using the following command: -
```
python train.py
```

You can now infer from the model from python using the following command: -
```
python inference.py
```
