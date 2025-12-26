import logging
from pathlib import Path

import fire
import numpy as np
import tritonclient.grpc as grpcclient
import uvicorn
from fastapi import FastAPI
from hydra import compose, initialize_config_dir
from pydantic import BaseModel
from transformers import AutoTokenizer


class InferRequest(BaseModel):
    context: str
    message: str


class InferResponse(BaseModel):
    probas: dict[str, float]


def create_app(config_name="config", **kwargs):
    config_path = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(
            config_name=config_name, overrides=[f"{k}={v}" for k, v in kwargs.items()]
        )

    app = FastAPI()
    triton_client = grpcclient.InferenceServerClient(url=cfg.serve.triton_url)
    tokenizer = AutoTokenizer.from_pretrained(cfg.serve.tokenizer)

    @app.post("/infer", response_model=InferResponse)
    async def infer(request: InferRequest):
        inputs = tokenizer(
            request.context,
            request.message,
            return_tensors="np",
            padding="max_length",
            max_length=512,
            truncation=True,
        )

        input_ids = grpcclient.InferInput("input_ids", [1, 512], "INT64")
        attention_mask = grpcclient.InferInput("attention_mask", [1, 512], "INT64")
        input_ids.set_data_from_numpy(inputs["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(inputs["attention_mask"].astype(np.int64))

        output = grpcclient.InferRequestedOutput("logits")
        response = triton_client.infer(
            model_name=cfg.serve.triton_model_name,
            inputs=[input_ids, attention_mask],
            outputs=[output],
        )

        logits = response.as_numpy("logits")[0]
        probs = np.exp(logits) / np.exp(logits).sum()

        return InferResponse(probas={"wait": float(probs[0]), "speak": float(probs[1])})

    return app, cfg


def serve(config_name="config", **kwargs):
    app, cfg = create_app(config_name, **kwargs)
    uvicorn.run(app, host=cfg.serve.api_host, port=cfg.serve.api_port)


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(serve)


if __name__ == "__main__":
    main()
