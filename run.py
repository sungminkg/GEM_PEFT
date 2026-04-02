import logging
import os
import random
import shutil
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from metrics import calculate_metric
from tasks import get_task
from utils import (
    DataCollatorWithPaddingAndNesting,
    Prediction,
    count_time,
    encode_prompt_eval,
    encode_prompt_train,
    forward_wrap_with_option_len,
)
from PEFT import Adapter, Bitfit, EntropyBasedMasking, GradWeightMasking, LoRA, RandomMasking


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("WANDB_MODE", "disabled")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULT_DIR = os.environ.get("RESULT_DIR", "results")
LOG_FILE_PATH = os.path.join(RESULT_DIR, "log.txt")
os.makedirs(RESULT_DIR, exist_ok=True)


@dataclass
class OurArguments(TrainingArguments):
    task_name: str = "SST2"

    fp16: bool = True
    bf16: bool = False

    num_train: int = 0
    num_dev: int = None
    num_eval: int = None
    num_train_sets: int = None
    train_set_seed: int = None
    result_file: str = None

    model_name: str = "facebook/opt-125m"
    max_length: int = 1024
    auto_device: bool = True

    only_train_option: bool = True
    train_as_classification: bool = False

    sampling: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    top_k: int = None
    top_p: float = 0.95
    max_new_tokens: int = 50
    eos_token: str = "\n"

    eval_batch_size: int = 8
    save_model: bool = False
    tag: str = ""

    lora: bool = False
    lora_alpha: int = 16
    lora_r: int = 8

    adapter: bool = False
    adapter_r: int = 8

    random_masking: bool = False
    masking_prob: float = 0.0

    gradient_masking: bool = False
    gradweight_masking: bool = False
    entropy_gradweight_masking: bool = False  # GEM: the main paper method.

    bitfit: bool = False
    fft: bool = False

    remove_unused_columns: bool = False


def parse_args():
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.local_rank > 0:
        logger.setLevel(level=logging.CRITICAL)

    if torch.cuda.is_available():
        if args.local_rank >= 0:
            torch.cuda.set_device(args.local_rank)
        cuda_empty_cache()
        cuda_reset_peak_memory_stats()

    logger.info(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class HFDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Framework:
    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer, self.pretrained_weights = self.initialize_model_and_tokenizer()
        self.load_model()

    def initialize_model_and_tokenizer(self):
        with count_time("Initializing model and tokenizer"):
            config = AutoConfig.from_pretrained(self.args.model_name)
            model_kwargs = {"config": config}
            if self.args.auto_device:
                model_kwargs["torch_dtype"] = torch.float32
            model = AutoModelForCausalLM.from_pretrained(self.args.model_name, **model_kwargs)

            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

            if "opt" in self.args.model_name.lower():
                tokenizer.bos_token_id = 0

            model_max_length_candidates = []
            if hasattr(config, "max_position_embeddings") and config.max_position_embeddings is not None:
                model_max_length_candidates.append(config.max_position_embeddings)
            if getattr(tokenizer, "model_max_length", None) and tokenizer.model_max_length < 100000:
                model_max_length_candidates.append(tokenizer.model_max_length)
            if model_max_length_candidates:
                supported_max_length = min(model_max_length_candidates)
                if self.args.max_length > supported_max_length:
                    logger.info(
                        "Reducing max_length from %d to %d to fit the selected model.",
                        self.args.max_length,
                        supported_max_length,
                    )
                    self.args.max_length = supported_max_length

            pretrained_weights = {
                name: param.clone().detach() for name, param in model.named_parameters()
            }
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            return model, tokenizer, pretrained_weights

    def _convert_samples(self, samples):
        data = []
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt_train(
                self.task,
                self.task.get_template(),
                [],
                sample,
                self.tokenizer,
                max_length=self.args.max_length,
                generation=self.task.generation,
                generation_with_gold=True,
                max_new_tokens=self.args.max_new_tokens,
            )
            if self.task.generation:
                correct_candidate_id = 0
            elif isinstance(sample.correct_candidate, list):
                correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)

            if self.args.train_as_classification:
                data.append(
                    [
                        {
                            "input_ids": encoded_candidates[idx],
                            "labels": correct_candidate_id,
                            "option_len": option_lens[idx],
                            "num_options": len(sample.candidates),
                        }
                        for idx in range(len(encoded_candidates))
                    ]
                )
            elif self.args.only_train_option:
                data.append(
                    {
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                        "option_len": option_lens[correct_candidate_id],
                    }
                )
            else:
                data.append(
                    {
                        "input_ids": encoded_candidates[correct_candidate_id],
                        "labels": encoded_candidates[correct_candidate_id],
                    }
                )
        return data

    def _convert_samples_for_backward(self, samples):
        data = []
        for sample in samples:
            encoded_candidates, _ = encode_prompt_train(
                self.task,
                self.task.get_template(),
                [],
                sample,
                self.tokenizer,
                max_length=self.args.max_length,
                generation=self.task.generation,
                generation_with_gold=True,
                max_new_tokens=self.args.max_new_tokens,
            )
            correct_candidate_id = 0 if self.task.generation else sample.candidates.index(sample.correct_candidate)
            data.append(
                {
                    "input_ids": encoded_candidates[correct_candidate_id],
                    "labels": encoded_candidates[correct_candidate_id],
                }
            )
        return data

    def backward_qv_pass(self):
        logger.info("Performing backward pass for q_proj/v_proj scoring.")
        train_samples = self.task.sample_subset("train", num=self.args.num_train)
        train_dataset = HFDataset(self._convert_samples_for_backward(train_samples))
        collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=collator,
        )

        self.model.train()
        gradients = {}
        for param in self.model.parameters():
            param.requires_grad = True

        for batch in tqdm(dataloader, desc="Backward Pass"):
            inputs = {key: val.to(self.model.device) for key, val in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss / self.args.gradient_accumulation_steps
            loss.backward()

            for name, param in self.model.named_parameters():
                if ("q_proj" in name or "v_proj" in name) and param.grad is not None:
                    if name not in gradients:
                        gradients[name] = param.grad.clone().detach()
                    else:
                        gradients[name] += param.grad.clone().detach()

            self.model.zero_grad(set_to_none=True)

        cuda_empty_cache()
        logger.info("Backward pass completed.")
        return gradients

    def load_model(self):
        cuda_reset_peak_memory_stats()
        cuda_synchronize()
        masking_start = time.time()

        if self.args.lora:
            LoRA(self.model, r=self.args.lora_r, alpha=self.args.lora_alpha)
        elif self.args.adapter:
            Adapter(self.model, r=self.args.adapter_r)
        elif self.args.random_masking:
            RandomMasking(self.model, masking_ratio=self.args.masking_prob)
        elif self.args.bitfit:
            Bitfit(self.model)
        elif self.args.gradient_masking:
            gradients = self.backward_qv_pass()
            GradWeightMasking(
                self.model,
                mask_mode="gradient",
                gradients=gradients,
                weights=self.pretrained_weights,
                masking_prob=self.args.masking_prob,
            )
        elif self.args.gradweight_masking:
            gradients = self.backward_qv_pass()
            GradWeightMasking(
                self.model,
                mask_mode="gradweight",
                gradients=gradients,
                weights=self.pretrained_weights,
                masking_prob=self.args.masking_prob,
            )
        elif self.args.entropy_gradweight_masking:
            gradients = self.backward_qv_pass()
            # `entropy_gradweight_masking` is the GEM method kept under its historical run name.
            EntropyBasedMasking(
                self.model,
                gradients=gradients,
                weights=self.pretrained_weights,
                masking_prob=self.args.masking_prob,
            )
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        cuda_synchronize()
        self.masking_runtime = time.time() - masking_start
        self.masking_memory = cuda_peak_memory_gb()
        logger.info("Masking runtime: %.2f seconds", self.masking_runtime)
        logger.info("Masking peak memory: %.2f GB", self.masking_memory)

        self.model.zero_grad()
        cuda_empty_cache()

        if torch.cuda.is_available():
            for param in self.model.parameters():
                param.data = param.data.half()

        for _, module in self.model.named_modules():
            if hasattr(module, "masking") and hasattr(module, "tunable_weight"):
                module.masking = module.masking.to(
                    dtype=module.tunable_weight.dtype,
                    device=module.tunable_weight.device,
                )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        percentage = (trainable_params / total_params) * 100
        logger.info(
            "Trainable params: %s / %s (%.6f%%)",
            trainable_params,
            total_params,
            percentage,
        )

    def forward(
        self,
        input_ids,
        attention_masks=None,
        option_len=None,
        generation=False,
        batch_size=None,
        num_of_candidates_arr=None,
    ):
        input_ids = torch.tensor(input_ids).to(self.model.device)
        attention_masks = torch.tensor(attention_masks).to(self.model.device)

        if generation:
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                do_sample=self.args.sampling,
                temperature=self.args.temperature,
                num_beams=self.args.num_beams,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                max_new_tokens=min(self.args.max_new_tokens, self.args.max_length - input_ids.size(1)),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(self.args.eos_token, add_special_tokens=False)[0],
                    self.tokenizer.eos_token_id,
                ],
            )
            return [
                self.tokenizer.decode(outputs[idx][input_ids[idx].size(0):], skip_special_tokens=True).strip()
                for idx in range(len(outputs))
            ]

        with torch.inference_mode():
            self.model.eval()
            logits = self.model(input_ids=input_ids, attention_mask=attention_masks).logits

        grouped_input_ids = []
        grouped_logits = []
        grouped_option_len = []
        start = 0
        for num_candidates in num_of_candidates_arr:
            end = start + num_candidates
            grouped_input_ids.append(input_ids[start:end])
            grouped_logits.append(logits[start:end])
            grouped_option_len.append(option_len[start:end])
            start = end

        selected_log_probs = []
        for batch_idx in range(batch_size):
            candidate_outputs = []
            for candidate_idx in range(num_of_candidates_arr[batch_idx]):
                label = grouped_input_ids[batch_idx][candidate_idx][1:]
                logit = grouped_logits[batch_idx][candidate_idx][:-1]
                log_probs = F.log_softmax(logit, dim=-1)
                token_log_probs = log_probs[torch.arange(len(label)).to(label.device), label]
                candidate_outputs.append(
                    token_log_probs.cpu().detach()[-grouped_option_len[batch_idx][candidate_idx]:]
                )
            selected_log_probs.append(candidate_outputs)

        return selected_log_probs

    def one_step_pred(self, eval_samples):
        batch_size = len(eval_samples)
        encoded_candidates, attention_masks, option_lens = encode_prompt_eval(
            self.task,
            self.task.get_template(),
            eval_samples,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens,
        )

        predictions = []
        if self.task.generation:
            output_texts = self.forward(
                encoded_candidates,
                attention_masks=attention_masks,
                generation=True,
                batch_size=batch_size,
            )
            for idx, sample in enumerate(eval_samples):
                predictions.append(
                    Prediction(
                        correct_candidate=sample.correct_candidate,
                        predicted_candidate=output_texts[idx],
                    )
                )
            return predictions

        num_of_candidates_arr = [len(sample.candidates) for sample in eval_samples]
        selected_log_probs = self.forward(
            encoded_candidates,
            attention_masks=attention_masks,
            option_len=option_lens,
            batch_size=batch_size,
            num_of_candidates_arr=num_of_candidates_arr,
        )
        scores = [[candidate.mean().item() for candidate in outputs] for outputs in selected_log_probs]

        for idx, sample in enumerate(eval_samples):
            if isinstance(sample.correct_candidate, list):
                correct_candidate_id = [sample.candidates.index(c) for c in sample.correct_candidate]
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)

            predictions.append(
                Prediction(
                    correct_candidate=correct_candidate_id,
                    predicted_candidate=int(np.argmax(scores[idx])),
                )
            )
        return predictions

    def evaluate(self, eval_samples):
        logger.info("Evaluating on %d validation samples", len(eval_samples))
        self.model.eval()
        cuda_empty_cache()

        batched_eval_samples = []
        for idx in range(len(eval_samples) // self.args.eval_batch_size):
            batched_eval_samples.append(
                eval_samples[idx * self.args.eval_batch_size:(idx + 1) * self.args.eval_batch_size]
            )
        if len(eval_samples) % self.args.eval_batch_size != 0:
            batched_eval_samples.append(eval_samples[-(len(eval_samples) % self.args.eval_batch_size):])

        predictions = []
        for batched_eval_sample in tqdm(batched_eval_samples):
            with torch.no_grad():
                predictions.extend(self.one_step_pred(batched_eval_sample))

        metric_name = getattr(self.task, "metric_name", "accuracy")
        final_score = calculate_metric(predictions, metric_name)
        metrics = {metric_name: final_score}
        save_experiment_log(
            {
                "model": self.args.model_name,
                "task": self.args.task_name,
                "mode": self.args.tag,
                "lr": self.args.learning_rate,
                "seed": self.args.train_set_seed,
                "final_accuracy": final_score,
                "masking_runtime_sec": getattr(self, "masking_runtime", "N/A"),
                "training_runtime_sec": getattr(self, "training_runtime", "N/A"),
                "masking_memory_gb": getattr(self, "masking_memory", "N/A"),
                "training_memory_gb": getattr(self, "training_memory", "N/A"),
            }
        )
        return metrics

    def train(self, train_samples, eval_samples):
        self.tokenizer.padding_side = "left"

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(self._convert_samples(train_samples))
            eval_dataset = HFDataset(self._convert_samples(eval_samples))

        if self.args.only_train_option:
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        data_collator = (
            DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8)
            if self.args.train_as_classification
            else DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        )

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = None
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. "
                "Change `--output_dir` or use `--overwrite_output_dir` to train from scratch.",
                last_checkpoint,
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        cuda_reset_peak_memory_stats()
        cuda_synchronize()
        train_start = time.time()
        trainer.train(resume_from_checkpoint=last_checkpoint)
        cuda_synchronize()

        self.training_runtime = time.time() - train_start
        self.training_memory = cuda_peak_memory_gb()
        logger.info("Training runtime: %.2f seconds", self.training_runtime)
        logger.info("Training peak memory: %.2f GB", self.training_memory)

        if not self.args.save_model and os.path.isdir(self.args.output_dir):
            shutil.rmtree(self.args.output_dir)

        self.model = trainer.model
        if self.args.only_train_option:
            self.model.forward = self.model.original_forward


def save_experiment_log(experiment_results):
    def fmt_time(value):
        if isinstance(value, (int, float)):
            return f"{value:.2f}s"
        return str(value)

    def fmt_num(value):
        if isinstance(value, (int, float)):
            return f"{value:.6f}"
        return str(value)

    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(
            f"MODEL={experiment_results['model']}, TASK={experiment_results['task']}, "
            f"MODE={experiment_results['mode']}, LR={experiment_results['lr']}, "
            f"SEED={experiment_results['seed']}, "
            f"MASKING_RUNTIME={fmt_time(experiment_results.get('masking_runtime_sec', 'N/A'))}, "
            f"TRAIN_RUNTIME={fmt_time(experiment_results.get('training_runtime_sec', 'N/A'))}, "
            f"MASKING_MEMORY={fmt_num(experiment_results.get('masking_memory_gb', 'N/A'))}, "
            f"TRAIN_MEMORY={fmt_num(experiment_results.get('training_memory_gb', 'N/A'))}, "
            f"ACC={fmt_num(experiment_results.get('final_accuracy', 'N/A'))}\n"
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=args.train_set_seed,
    )

    framework = Framework(args, task)

    for train_set_id, train_samples in enumerate(train_sets):
        train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        if args.num_dev is not None:
            dev_samples = train_samples[-args.num_dev:]
            train_samples = train_samples[:-args.num_dev]
        else:
            dev_samples = None

        logger.info(
            "Train set %d has %d training samples, %d dev samples, and %d eval samples",
            train_set_id,
            len(train_samples),
            len(dev_samples) if dev_samples is not None else 0,
            len(eval_samples),
        )

        framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
        metrics = framework.evaluate(eval_samples)

        logger.info("===== Train set %d =====", train_set_seed)
        logger.info(metrics)


if __name__ == "__main__":
    main()
