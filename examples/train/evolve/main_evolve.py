"""SkyRL training entrypoint for EvolveGenerator advisor RL."""

import sys

import ray
from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from scaleevolve.training.evolve_generator import EvolveGenerator, EvolveGeneratorConfig
from scaleevolve.training.dataset import EvolveTaskDataset


EvolveConfig = make_config(generator_cls=EvolveGeneratorConfig)


class EvolveExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return EvolveGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
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
def skyrl_entrypoint(cfg):
    exp = EvolveExp(cfg)
    exp.run()


def main() -> None:
    cfg = EvolveConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
