from fastapi import FastAPI, Request, HTTPException
import requests
import os
import time
import uuid
import logging
import json
from tabulate import tabulate

# FastAPI 앱 초기화
app = FastAPI()

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# 환경 변수에서 필요한 값 가져오기
SOFTWARE_URL = os.getenv("SOFTWARE_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not SOFTWARE_URL:
    logger.error("Software URL is not set. Please set the SOFTWARE_URL environment variable.")
    raise SystemExit("SOFTWARE_URL is required.")

if not PROJECT_ID:
    logger.error("Project ID is not set. Please set the PROJECT_ID environment variable.")
    raise SystemExit("PROJECT_ID is required.")

if not BEARER_TOKEN:
    logger.error("Bearer token is not set. Please set the BEARER_TOKEN environment variable.")
    raise SystemExit("BEARER_TOKEN is required.")

# FastAPI 라우트: 모델 리스트 가져오기
@app.get("/v1/models")
async def fetch_models():
    logger.info("Fetching model list.")
    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response = requests.get(SOFTWARE_URL.replace("/text/generation", "/foundation_model_specs"), headers=headers)
        response.raise_for_status()
        models_data = response.json()
        logger.debug(f"Models fetched: {json.dumps(models_data, indent=4)}")

        # OpenAI 호환 형식으로 변환
        models = []
        for model in models_data.get("resources", []):
            # model_limits가 없을 경우 기본값 설정
            model_limits = model.get("model_limits", {"max_output_tokens": 2000, "max_sequence_length": 4096})
            models.append({
                "id": model["model_id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ibm",
                "description": model.get("short_description", ""),
                "max_tokens": model_limits.get("max_output_tokens", 2000),
            })

        return {"data": models}
    except Exception as err:
        logger.error(f"Error fetching models: {err}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {err}")

# FastAPI 라우트: 텍스트 생성 엔드포인트
@app.post("/v1/chat/completions")
async def watsonx_completions(request: Request):
    logger.info("Received a Watsonx completion request.")

    try:
        # 요청 데이터 파싱
        request_data = await request.json()
        logger.debug(f"Received request data: {json.dumps(request_data, indent=4)}")

        # 'messages' 필드 처리
        messages = request_data.get("messages", [])
        if not messages or not isinstance(messages, list):
            logger.error("The 'messages' field is empty or invalid.")
            raise HTTPException(status_code=400, detail="The 'messages' field is empty or invalid.")

        # 마지막 메시지의 content를 추출하여 prompt로 사용
        prompt = messages[-1].get("content", "").strip()
        if not prompt:
            logger.error("The 'content' field in the last message is empty.")
            raise HTTPException(status_code=400, detail="The 'content' field in the last message cannot be empty.")
    except Exception as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON request body")

    # Watsonx API 요청 payload 생성
    model_id = request_data.get("model", "meta-llama/llama-3-1-8b-instruct")
    max_tokens = request_data.get("max_tokens", 200)
    min_tokens = request_data.get("min_tokens", 0)
    repetition_penalty = request_data.get("repetition_penalty", 1)
    stop_sequences = request_data.get("stop_sequences", [])

    payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": max_tokens,
            "min_new_tokens": min_tokens,
            "stop_sequences": stop_sequences,
            "repetition_penalty": repetition_penalty
        },
        "model_id": model_id,
        "project_id": PROJECT_ID
    }

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Watsonx API 호출
    try:
        logger.debug(f"Sending request to Watsonx API with payload: {json.dumps(payload, indent=4)}")
        response = requests.post(SOFTWARE_URL, headers=headers, json=payload)
        response.raise_for_status()
        watsonx_response = response.json()
        logger.debug(f"Watsonx API response: {json.dumps(watsonx_response, indent=4)}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Error calling Watsonx API: {err}")
        raise HTTPException(status_code=500, detail=f"Error calling Watsonx API: {err}")

    # OpenAI API 호환 응답 형식으로 변환
    results = watsonx_response.get("results", [])
    if results and "generated_text" in results[0]:
        generated_text = results[0]["generated_text"]
        stop_reason = results[0].get("stop_reason", "stop")
        input_token_count = results[0].get("input_token_count", 0)
        generated_token_count = results[0].get("generated_token_count", 0)
    else:
        logger.error("Watsonx API response is missing 'generated_text'.")
        raise HTTPException(status_code=500, detail="Invalid Watsonx API response.")

    # OpenAI 호환 응답 생성
    return {
        "id": f"chatcmpl-{str(uuid.uuid4())[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": f"fp_{str(uuid.uuid4())[:12]}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text.strip()
                },
                "logprobs": None,
                "finish_reason": stop_reason
            }
        ],
        "service_tier": "default",
        "usage": {
            "prompt_tokens": input_token_count,
            "completion_tokens": generated_token_count,
            "total_tokens": input_token_count + generated_token_count,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        }
    }

