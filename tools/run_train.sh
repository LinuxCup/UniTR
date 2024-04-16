
# python train.py --cfg_file ./cfgs/nuscenes_models/unitr.yaml  --pretrained_model ../unitr_pretrain.pth

bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/unitr.yaml --sync_bn --pretrained_model ../unitr_pretrain.pth --logger_iter_interval 300