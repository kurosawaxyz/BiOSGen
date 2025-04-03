python test.py --dataroot ../pix2pix/data/ --direction BtoA --model pix2pix --name tumor_pix2pix --use_wandb && \
echo "Testing started."

# Move the trained model to the checkpoints directory
mkdir -p ../checkpoints && \
mv ../pytorch-CycleGAN-and-pix2pix/checkpoints/tumor_pix2pix ../checkpoints/ && \
echo "Model moved to checkpoints directory."