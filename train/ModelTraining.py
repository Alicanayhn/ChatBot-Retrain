import mlflow
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer 
# import torch

# fp16_mode = torch.cuda.is_available()

def load_distilgpt2_model(model_name="distilbert/distilgpt2"):
    # try:
    #     MODEL_URI = "models:/distilgpt2-chatbot@prod"
    #     model = mlflow.pytorch.load_model(MODEL_URI)
    #     tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    #     return model, tokenizer
    # except:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer

def finetune(model, tokenizer, tokenized_dataset, output_dir="./distilgpt2-finetuned"):
    print("Fine Tuning Başlıyor")
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50,
        fp16=False,                      
        save_total_limit=2,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        # eval_dataset=tokenized_dataset.select(range(0, 10)),
        tokenizer=tokenizer
    )

    with mlflow.start_run():
        trainer.train()

        # example_input = tokenizer("Merhaba!", return_tensors="np")["input_ids"]

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
        )

        mlflow.log_params({
            "epochs": 3,
            "batch_size": 2,
            "fp16": False
        })

        # eval_result = trainer.evaluate()

        # for metric_name, metric_value in eval_result.items():
        #     if isinstance(metric_value, torch.Tensor):
        #         metric_value = metric_value.item()
        #     mlflow.log_metric(metric_name, metric_value)

        mlflow.log_metric("token_count", len(tokenized_dataset))

    # print(f"Model burada kaydedildi: {output_dir}")
