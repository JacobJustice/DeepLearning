# How to use

To train a new model, step into the respective models directory and run `python train.py`.

Your new models will be saved every 2 epochs in the models directory.

To view the results of your model, run `python generate_img.py -load_path <model_filename>`

I have left my best DCGAN in `dcgan/models` and my best WGAN in `wgan/models`. To view those, simply run `python generate_img.py` in their respective directories.