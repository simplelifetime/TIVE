from utils import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", add_help=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_base", type=str, default="")
    parser.add_argument("--output_file", type=str, default='output_chunks')
    args = parser.parse_args()
    
    target_dir = args.output_file
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Directory Created:", target_dir)
    else:
        print("Directory Existed:", target_dir)
        
    ## loading data and model, setting conversation format and project dimension
    data = json.load(open(args.input_file, 'r'))
    tokenizer, model, image_processor, _ = load_pretrained_model_lora(model_path=args.model_path, model_base=args.model_base, model_name='llava_lora')
    tokenizer.model_max_length = 2048
    for n, p in model.named_parameters():
        if 'lora' in n and 'self_attn' in n:
            p.requires_grad = True
    conv_mode = "llava_v1"
    
    grad_dim = 134217728
    proj_dim = 8192
    device = model.device 
    dtype = model.dtype

    from trak.projectors import BasicProjector, CudaProjector, ProjectionType

    def get_trak_projector(device: torch.device):
        """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
        try:
            num_sms = torch.cuda.get_device_properties(
                device.index).multi_processor_count
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(torch.zeros(
                8, 1_000, device=device), 512, 0, num_sms)
            projector = CudaProjector
            print("Using CudaProjector")
        except:
            projector = BasicProjector
            print("Using BasicProjector")
        return projector
    projector = get_trak_projector(device)

    proj = projector(grad_dim=grad_dim,
                            proj_dim=proj_dim,
                            seed=0,
                            proj_type=ProjectionType.rademacher,
                            device=device,
                            dtype=dtype,
                            block_size=128,
                            max_batch_size=16)
    
    chunk_data = get_chunk(data, args.num_chunks, args.chunk_idx)
    all_grads = []
    all_norms = []
    for cd in tqdm(get_chunks_ind(chunk_data, 16)):
        grads, norms = compute_projected_gradients(cd, model, tokenizer, image_processor, proj)
        all_grads.append(grads.cpu())
        all_norms.extend(norms)

    
    torch.save(all_grads, f"{target_dir}/output_{args.chunk_idx}")
    json.dump(all_norms, open(f"{target_dir}/output_norm_{args.chunk_idx}.json", 'w'), indent=4)