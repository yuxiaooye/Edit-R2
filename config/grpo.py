import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2
    config.val_only = False

    
    config.vae_donot_sample = True
    config.gen_use_vit = False
    config.cfg_renorm_type = "text_channel"
    config.text_key_scale = 1.0
    config.train.use_velocity_kl = False

    return config


def multiturn():
    config = compressibility()

    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.sample.cfg_img_scale = 2.0
    config.train.cfg = True     # No effect for BAGEL, always use cfg in code.
    config.train.ema = False
    config.use_lora = True

    config.resolution = 512
    config.sample.test_batch_size = 1 

    config.train.num_inner_epochs = 1
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.mixed_precision = "bf16"

    config.sample.noise_level = 1.3

    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps//2)


    config.sample.same_latent = True
    config.sample.global_std = True
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True

    return config


def editr2():
    config = multiturn()

    config.save_dir = "logs/edit-r2"
    config.run_name = "edit-r2"
    config.num_turns = 3

    config.pretrained.model = "/path/to/BAGEL-7B-MoT"

    # 24 GPUs accross 4 nodes
    gpu_number = 24
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt)) # 2
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 1

    config.reward_fn = {
        "edival_client_if": 0.3,
        "edival_client_cc": 0.3,
        "edival_client_cu": 0.4, # NOTE
    }

    config.dataset = os.path.join(os.getcwd(), "dataset/cu")

    config.edival_if_num_votes = 4
    config.edival_ga_num_votes = 4    
    config.edival_ga_temperature = 0.6

    config.vae_transform = (1024, 512, 16)
    config.vit_transform = (980, 224, 14)

    config.use_vit = True
    config.gen_use_vit = True # NOTE
    config.use_ic_cot = True # NOTE
    config.use_ic_cot_grpo_loss = True # NOTE
    config.nonmarkov_mode = 'cu' # NOTE
    config.pe_max_token_n = 150

    config.train.beta = 0.01

    config.eval_freq = 20
    config.save_freq = 20

    return config



def get_config(name):
    return globals()[name]()
