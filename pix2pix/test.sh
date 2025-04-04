# Change directory to the cloned repository
cd pytorch-CycleGAN-and-pix2pix || exit

python test.py --dataroot ../pix2pix/data/ --direction BtoA --model pix2pix --name tumor_pix2pix  && \

# Print a message when testing is finished
echo "Testing finished."
