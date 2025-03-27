import numpy as np
import torch

from i2iTranslation.base_model import BaseModel
from i2iTranslation import network  
from i2iTranslation.local_loss import GANLoss, PatchNCELoss
from i2iTranslation.global_loss import TileLevelCriterion

class i2iTranslationModel(BaseModel):
    def __init__(self, args):
        """
        Parameters:
            args -- stores all the experiment information
        """
        BaseModel.__init__(self, args)
        self.set_params(args)

        # define generator net
        self.netG = network.define_G(args).to(self.device)
        self.netF = network.define_F(args).to(self.device)

        if self.is_train:
            # define discriminator net
            self.netD = network.define_D(args).to(self.device)

            # define loss functions
            self.criterionGAN = GANLoss(self.gan_mode, device=self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = PatchNCELoss(args, self.device)

            # define optimizer
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=self.lr_G,
                betas=(self.beta1, self.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.lr_D,
                betas=(self.beta1, self.beta2)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # image-level style and content modules
            if self.is_image_match:
                self.criterionImage = TileLevelCriterion(args)
                self.optimizer_I = torch.optim.Adam(
                    self.netG.parameters(),
                    lr=self.lr_I,
                    betas=(self.beta1, self.beta2)
                )
                self.optimizers.append(self.optimizer_I)
        
        # load data to device
        for model_name in self.model_names:
            net = getattr(self, 'net' + model_name)
            setattr(self, 'net' + model_name, net.to(self.device))

            
    def set_params(self, args):
        self.input_nc = args.train.data.input_nc
        self.output_nc = args.train.data.output_nc

        # Model
        self.netF_mode = args.train.model.projector.netF

        # Neighborhood objective (GAN + NCE loss) params
        self.gan_mode = args.train.params.loss.gan_loss.gan_mode
        self.nce_idt = args.train.params.loss.gan_loss.nce_idt
        self.lambda_NCE = args.train.params.loss.gan_loss.lambda_NCE
        self.nce_layers = [int(i) for i in args.train.params.loss.gan_loss.nce_layers.split(',')]
        self.nce_num_patches = args.train.params.loss.gan_loss.nce_num_patches
        self.flip_equivariance = args.train.params.loss.gan_loss.flip_equivariance

        # Optimizer params
        self.lr_G = args.train.params.optimizer.lr.lr_G
        self.lr_D = args.train.params.optimizer.lr.lr_D
        self.beta1 = args.train.params.optimizer.params.beta1
        self.beta2 = args.train.params.optimizer.params.beta2

        # Global objective (Style & Content loss) params
        self.is_image_match = args.train.params.is_image_match
        if self.is_image_match:
            self.lambda_con = args.train.params.loss.sc_loss.lambda_con
            self.lambda_sty = args.train.params.loss.sc_loss.lambda_sty
            self.lr_I = args.train.params.optimizer.lr.lr_I

        # models to save/load to the disk
        if self.is_train:
            self.model_names = ['G', 'F', 'D']
            
        else:
            self.model_names = ['G']

        # specify the training losses to #print out. training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'NCE_A', 'NCE_B', 'D_real', 'D_fake']


        # specify the images to save/display. training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        if self.is_train:
            self.visual_names += ['real_B']
            if self.nce_idt:
                self.visual_names += ['idt_B']

    def set_input(self, input):
        """Ensure input tensors require gradients."""
        self.real_A = input['src'].to(self.device)
        self.real_B = input['dst'].to(self.device)
        
        # Ensure gradients
        self.real_A.requires_grad = True
        self.real_B.requires_grad = True
        #print(f"self.real_A requires grad: {self.real_A.requires_grad}")
        #print(f"self.real_B requires grad: {self.real_B.requires_grad}")



    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        self.forward()                     # compute fake images: G(A)
        if self.is_train:
            #print("self training")
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                  # calculate gradients for G
            if self.lambda_NCE > 0.0:
                #print("Initializing optimizer_F...")

                # Check if netF has parameters
                param_list = list(self.netF.parameters())
                #print(f"Before create_mlp - netF parameters: {len(param_list)}")

                if not self.netF.mlp_init:
                    #print("Creating netF MLP layers...")
                    self.netF.create_mlp(self.fake_B)  # Use features from netG

                param_list = list(self.netF.parameters())  # Check after create_mlp
                #print(f"After create_mlp - netF parameters: {len(param_list)}")

                if len(param_list) == 0:
                    raise RuntimeError("netF still has no trainable parameters!")

                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(),
                    lr=self.lr_G,
                    betas=(self.beta1, self.beta2)
                )
                self.optimizers.append(self.optimizer_F)

                #print(f"self.real_B requires grad: {self.real_B.requires_grad}")
                #print(f"self.fake_B requires grad: {self.fake_B.requires_grad}")


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #print("Starting forward pass...")
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt and self.is_train else self.real_A
        if self.flip_equivariance:
            self.flipped_for_equivariance = self.is_train and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        #print("Forward pass - real shape:", self.real.shape)
        #print("starting forward pass - netG")
        self.fake = self.netG(self.real, encode_only=False)
        #print("fake shape model forward", self.fake.shape)

        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        #print("finished forward pass - netG")

    def compute_D_loss(self):
        """Ensure netD requires gradients"""
        for param in self.netD.parameters():
            param.requires_grad = True  # Enable gradients

        self.loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        self.loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)

        # Combine loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # Debugging: Check if loss requires grad
        #print(f"loss_D_real requires grad: {self.loss_D_real.requires_grad}")
        #print(f"loss_D_fake requires grad: {self.loss_D_fake.requires_grad}")
        #print(f"Final loss_D requires grad: {self.loss_D.requires_grad}")

        return self.loss_D



    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""

        # GAN loss: D(G(A))
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)

        # NCE loss: NCE(A, G(A))
        if self.lambda_NCE > 0.0:
            #print("Starting NCE loss calculation...")
            self.loss_NCE_A = self.compute_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE_A = 0.0

        # NCE-IDT loss: NCE(B, G(B))
        if self.nce_idt and self.lambda_NCE > 0.0:
            self.loss_NCE_B = self.compute_NCE_loss(self.real_B, self.idt_B)
        else:
            self.loss_NCE_B = 0
        loss_NCE = self.lambda_NCE * (self.loss_NCE_A + self.loss_NCE_B) * 0.5

        self.loss_G = self.loss_G_GAN + loss_NCE 
        return self.loss_G

    def compute_NCE_loss(self, src, tgt):

        feat_q, patch_ids_q = self.netG(tgt, num_patches=self.nce_num_patches, encode_only=True)

        if self.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k, _ = self.netG(src, num_patches=self.nce_num_patches, encode_only=True, patch_ids=patch_ids_q)

        #print("feat_q shape:", feat_q[0].shape)
        #print("feat_q type:", type(feat_q))  # Should be a list of tensors

        #print("feat_k shape:", feat_k[0].shape)
        feat_k_pool = self.netF([f for f in feat_k if len(f.shape) == 4])
        feat_q_pool = self.netF([f for f in feat_q if len(f.shape) == 4])

        #print("finished NCE loss calculation")

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / len(self.nce_layers)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G and F
        self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        if self.netF_mode == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.netF_mode == 'mlp_sample':
            self.optimizer_F.step()

    def set_input_image(self, input):
        if self.is_train:
            # move images to device
            self.real_A_img = input['src_real'].to(self.device)
            self.real_B_img = input['dst_real'].to(self.device)
            self.fake_B_img = input['dst_fake'].to(self.device)

    def optimize_parameters_image(self):
        self.set_requires_grad(self.netD, False)                   # D require no gradients when optimizing G

        self.optimizer_I.zero_grad()                                           # set G gradients to zero

        # compute style and content losses at image-level
        loss_content_B_fake, loss_style_B_fake = self.criterionImage(self.real_A_img, self.real_B_img, self.fake_B_img)
        self.loss_content_B_fake = self.lambda_con * loss_content_B_fake
        self.loss_style_B_fake = self.lambda_sty * loss_style_B_fake

        # calculate gradients for G_A
        self.loss_I = 0.5 * (self.loss_content_B_fake + self.loss_style_B_fake)
        self.loss_I.backward()                                     # back propagate for G_A
        self.optimizer_I.step()                                    # update G_A's weights

        return {'image' : self.loss_I.detach().item(),
                'content_B_fake' : self.loss_content_B_fake.detach().item(),
                'style_B_fake': self.loss_style_B_fake.detach().item()}