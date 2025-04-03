python test.py --dataroot ../cyclegan/data/testA --name tumor_cyclegan --model test --no_dropout && \
echo "Testing started."

# Move the trained model to the checkpoints directory
mkdir -p ../checkpoints && \
mv ../pytorch-CycleGAN-and-pix2pix/checkpoints/tumor_cyclegan ../checkpoints/ && \
echo "Model moved to checkpoints directory."