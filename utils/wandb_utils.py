from typing import Dict, Optional
import os
import wandb

__all__ = ["set_wandb"]


def set_wandb(opt: Dict, local_rank: int = 0, force_mode: Optional[str] = None) -> str:
    if local_rank != 0:
        return ""

    # opt = opt
    save_dir = os.path.join(opt["output_dir"], opt["wandb"]["name"]) # root save dir

    wandb_mode = opt["wandb"]["mode"].lower()
    if force_mode is not None:
        wandb_mode = force_mode.lower()
    if wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"WandB mode {wandb_mode} invalid.")

    os.makedirs(save_dir, exist_ok=True)

    wandb_project = opt["wandb"]["project"]
    wandb_entity = opt["wandb"]["entity"]
    wandb_name = opt["wandb"]["name"]
    wandb_id = opt["wandb"].get("id", None)
    wandb_notes = opt["wandb"].get("notes", None)
    wandb_tags = opt["wandb"].get("tags", None)
    if wandb_tags is None:
        wandb_tags = [opt["dataset"]["data_type"], ]

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        dir=save_dir,
        resume="allow",
        mode=wandb_mode,
        id=wandb_id,
        notes=wandb_notes,
        tags=wandb_tags,
        config=opt,
    )
    wandb_path = wandb.run.dir if (wandb_mode != "disabled") else save_dir
    return wandb_path
