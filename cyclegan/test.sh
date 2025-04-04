# Copy the checkpoint file
cp pytorch-CycleGAN-and-pix2pix/checkpoints/horse2zebra/latest_net_G_A.pth checkpoints/horse2zebra/latest_net_G.pth

# Run the test script
python pytorch-CycleGAN-and-pix2pix/test.py --dataroot cyclegan/data/testA --name tumor_cyclegan --model test --no_dropout && \

# Print a message when testing is finished
echo "Testing finished."
