python3 make_fussy_models.py models ./backbones/VGG16_base.h5 100
python3 extract_features.py models
python3 train_models.py models
python3 test_models.py models
