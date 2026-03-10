CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/CARGO/Vit_base.yml --eval-only MODEL.WEIGHTS ./logs/CARGO/ViT_base/model_best.pth
python demo/visualize_tSNE.py --config-file ./CARGO/Vit_base.yml --parallel --vis-label  --output logs/t-sne --dataset-name cargo --opts MODEL.WEIGHTS ./logs/CARGO/ViT_base/model_best.pth
python tools/visualize_tsne.py \
  --config-file configs/Market1501/bagtricks_R50.yml \
  --dataset-name Market1501 \
  --output ./vis_tsne
