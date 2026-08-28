import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    MetricWindow,
    SkipBatchSampler,
    evaluate_causal_lm,
    experiment_config,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    reduce_sums,
    safe_swanlab_log,
    setup_seed,
)


def build_parser(description, defaults, use_lora=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save_dir", default="../out")
    parser.add_argument("--save_weight", default=defaults["save_weight"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=defaults["accumulation_steps"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10, help="optimizer steps")
    parser.add_argument("--save_interval", type=int, default=500, help="optimizer steps")
    parser.add_argument("--eval_interval", type=int, default=defaults["eval_interval"], help="optimizer steps")
    parser.add_argument("--eval_batches", type=int, default=50, help="0 evaluates the whole validation set")
    parser.add_argument("--max_steps", type=int, default=0, help="maximum optimizer steps; 0 is unlimited")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=defaults["max_seq_len"])
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--data_path", default=defaults["data_path"])
    parser.add_argument("--val_data_path", default=defaults.get("val_data_path"))
    parser.add_argument("--from_weight", default=defaults["from_weight"])
    parser.add_argument("--from_resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true")
    parser.add_argument("--swanlab_mode", "--wandb_mode", dest="swanlab_mode", choices=["cloud", "local", "offline", "disabled"], default="cloud")
    parser.add_argument("--swanlab_logdir", "--wandb_logdir", dest="swanlab_logdir", default=None)
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", default=defaults["project"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--use_compile", type=int, choices=[0, 1], default=0)
    if use_lora:
        parser.add_argument("--lora_rank", type=int, default=16)
    return parser


def run_lm_training(dataset_class, description, defaults, use_lora=False):
    args = build_parser(description, defaults, use_lora).parse_args()
    lora_compile_disabled = use_lora and bool(args.use_compile)
    if lora_compile_disabled:
        args.use_compile = 0
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(42 + rank)
    if lora_compile_disabled:
        Logger("LoRA uses patched attention forwards; torch.compile disabled")
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume else None
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda", dtype=dtype)

    tracker = None
    if args.use_swanlab and is_main_process():
        import swanlab

        run_id = (ckp_data or {}).get("swanlab_id") or (ckp_data or {}).get("wandb_id")
        tracker = swanlab
        tracker.init(
            project=args.swanlab_project,
            name=args.run_name or f"OceanHeart-{defaults['stage']}-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
            id=run_id,
            resume="must" if run_id else None,
            mode=args.swanlab_mode,
            logdir=args.swanlab_logdir,
            config=experiment_config(args),
        )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if use_lora:
        from model.model_lora import apply_lora

        targets = apply_lora(model, args.lora_rank)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora." in name
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        Logger(f"LoRA targets: {targets}, trainable: {trainable / 1e6:.3f}M ({100 * trainable / total:.2f}%)")
    train_ds = dataset_class(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_ds = dataset_class(args.val_data_path, tokenizer, max_length=args.max_seq_len) if args.val_data_path else None
    if val_ds is not None and hasattr(val_ds, "deterministic"):
        val_ds.deterministic = True
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler("cuda", enabled=device_type == "cuda" and args.dtype == "float16")
    optimizer = optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)

    start_epoch = start_step = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)
    if args.use_compile:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    def save_checkpoint(epoch, step):
        if not is_main_process():
            return
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        suffix = "_moe" if lm_config.use_moe else ""
        path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{suffix}.pth"
        if use_lora:
            from model.model_lora import save_lora, save_merged_lora

            save_lora(raw_model, path)
            save_merged_lora(raw_model, f"{args.save_dir}/{args.save_weight}_merged_{lm_config.hidden_size}{suffix}.pth")
        else:
            state = {key: value.half().cpu() for key, value in raw_model.state_dict().items()}
            temp = path + ".tmp"
            torch.save(state, temp)
            os.replace(temp, path)
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=step,
            wandb=tracker,
            save_dir="../checkpoints",
        )
        if not use_lora:
            del state

    def validate(global_update, full=False):
        nonlocal tracker
        if not val_loader:
            return
        metrics = evaluate_causal_lm(
            model, val_loader, args.device, autocast_ctx, max_batches=0 if full else args.eval_batches
        )
        if metrics:
            metrics["perplexity"] = math.exp(min(metrics["logits_loss"], 20))
            tracker = safe_swanlab_log(
                tracker, {f"val/{key}": value for key, value in metrics.items()}, step=global_update
            )
            Logger(f"Validation step {global_update}: loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")

    if val_loader and not ckp_data:
        validate(0)

    stop = False
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=device_type == "cuda")
        iters = len(loader) + skip
        window, window_tokens, window_start = MetricWindow(), 0, time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, labels) in enumerate(loader, start=skip + 1):
            input_ids, labels = input_ids.to(args.device), labels.to(args.device)
            lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast_ctx:
                output = model(input_ids, labels=labels)
                logits_loss, aux_loss = output.loss, output.aux_loss
                loss = (logits_loss + aux_loss) / args.accumulation_steps
            scaler.scale(loss).backward()
            valid_tokens = labels[..., 1:].ne(-100).sum().item()
            window.add("logits_loss", logits_loss, valid_tokens)
            window.add("aux_loss", aux_loss, input_ids.size(0))
            window_tokens += valid_tokens

            final_micro_step = step == iters
            if step % args.accumulation_steps and not final_micro_step:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_update = math.ceil((epoch * iters + step) / args.accumulation_steps)
            window.add("grad_norm", grad_norm)
            window.add("grad_clipped", float(grad_norm > args.grad_clip))

            should_log = global_update % args.log_interval == 0 or final_micro_step
            if should_log:
                metrics = window.means(args.device)
                total_tokens = reduce_sums([window_tokens], args.device)[0]
                elapsed = max(time.time() - window_start, 1e-6)
                metrics["loss"] = metrics["logits_loss"] + metrics["aux_loss"]
                metrics["perplexity"] = math.exp(min(metrics["logits_loss"], 20))
                metrics["learning_rate"] = lr
                metrics["tokens_per_second"] = total_tokens / elapsed
                tracker = safe_swanlab_log(
                    tracker, {f"train/{key}": value for key, value in metrics.items()}, step=global_update
                )
                Logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) update={global_update}, "
                    f"loss={metrics['loss']:.4f}, lr={lr:.8f}, tokens/s={metrics['tokens_per_second']:.0f}"
                )
                window_tokens, window_start = 0, time.time()

            if val_loader and global_update % args.eval_interval == 0:
                validate(global_update)
            if global_update % args.save_interval == 0:
                save_checkpoint(epoch, step)
            if args.max_steps and global_update >= args.max_steps:
                save_checkpoint(epoch, step)
                stop = True
                break

        if not stop:
            save_checkpoint(epoch, iters)
            validate(math.ceil(((epoch + 1) * iters) / args.accumulation_steps), full=True)
        start_step = 0
        if stop:
            break

    if tracker and hasattr(tracker, "finish"):
        tracker.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
