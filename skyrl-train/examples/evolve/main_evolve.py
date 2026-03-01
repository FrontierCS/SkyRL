"""SkyRL training entrypoint for EvolveGenerator advisor RL."""

import ray
import hydra
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray

from scaleevolve.training.evolve_generator import EvolveGenerator
from scaleevolve.training.dataset import EvolveTaskDataset


class EvolveExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return EvolveGenerator(
            generator_cfg=cfg.generator,
            evolve_cfg=cfg.evolve,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        dataset = EvolveTaskDataset(data_files=self.cfg.data.train_data)
        assert len(dataset) >= self.cfg.trainer.train_batch_size, (
            f"Dataset size {len(dataset)} < train_batch_size {self.cfg.trainer.train_batch_size}"
        )
        return dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return EvolveTaskDataset(data_files=self.cfg.data.val_data)
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = EvolveExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
