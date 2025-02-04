from fastapi import FastAPI, Request, HTTPException
import requests
import os
import time
import uuid
import logging
import json

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 환경 변수 설정
BASE_URL = os.getenv("BASE_URL")
PROJECT_ID = os.getenv("PROJECT_ID", "")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
MILVUS_SEARCH_API_URL = os.getenv("MILVUS_SEARCH_API_URL", "")
SOFTWARE_URL = f"{BASE_URL}/ml/v1/deployments/9d28fc1b-1958-4647-84ed-d06c008f2b50/text/generation?version=2023-05-29"
TOKEN_URL = f"{BASE_URL}/icp4d-api/v1/authorize"

TOKEN_CREDENTIALS = {
    "username": USERNAME,
    "password": PASSWORD
}

if not BASE_URL or not USERNAME or not PASSWORD:
    logger.error("필수 환경 변수가 설정되지 않았습니다.")
    raise SystemExit("BASE_URL, USERNAME, PASSWORD를 설정해주세요.")

def get_bearer_token():
    """IBM watsonx.ai API에 대한 Bearer Token을 가져오는 함수"""
    try:
        response = requests.post(TOKEN_URL, headers={
            "cache-control": "no-cache",
            "Content-Type": "application/json"
        }, json=TOKEN_CREDENTIALS, verify=False)  # SSL 인증 무시 (-k 옵션)

        response.raise_for_status()
        token_data = response.json()
        return token_data.get("token")
    except Exception as e:
        logger.error(f"Bearer Token 요청 실패: {e}")
        return None

@app.get("/v1/models")
async def fetch_models():
    logger.info("Fetching model list.")
    BEARER_TOKEN = get_bearer_token()
    if not BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="watsonx.ai Bearer Token을 가져오는 데 실패했습니다.")
    
    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response = requests.get(f"{BASE_URL}/ml/v1/foundation_model_specs?version=2023-05-29", headers=headers)
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

@app.post("/v1/chat/completions")
async def watsonx_completions(request: Request):
    BEARER_TOKEN = get_bearer_token()
    if not BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="watsonx.ai Bearer Token을 가져오는 데 실패했습니다.")

    try:
        request_data = await request.json()
        logger.debug(f"요청 데이터: {json.dumps(request_data, indent=4)}")

        messages = request_data.get("messages", [])
        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="메시지 필드가 비어있거나 올바르지 않습니다.")

        query = messages[-1].get("content", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="질문 내용이 비어있습니다.")
    except Exception as e:
        logger.error(f"잘못된 요청: {e}")
        raise HTTPException(status_code=400, detail="잘못된 JSON 요청입니다.")

    # Milvus 검색 API 호출
    try:
        search_response = requests.post(MILVUS_SEARCH_API_URL, json={"query": query})
        search_response.raise_for_status()
        search_data = search_response.json()
        milvus_results = search_data.get("results", [])

        # 검색 결과에서 텍스트, 이미지, 테이블 데이터 추출
        context_list = []
        image_list = []
        table_list = []

        for res in milvus_results:
            if res["type"] == "text":
                chunk = res["data"].get("chunk", {})
                context_text = f"{chunk.get('대분류', '')} - {chunk.get('중분류', '')} - {chunk.get('소분류', '')} - {chunk.get('본문', '')}"
                if context_text.strip():
                    context_list.append(context_text)
            elif res["type"] == "image":
                image_list.append(res["data"].get("path", ""))
            elif res["type"] == "table":
                table_html = res["data"].get("html", "")
                if table_html.strip():
                    table_list.append(table_html)

        # 검색 결과를 LLM 프롬프트에 추가
        context_str = "\n".join(context_list)
        if image_list:
            image_str = "\nImage to refer:\n" + "\n".join(image_list)
        else:
            image_str = "N/A"

        if table_list:
            table_str = "\nTable data:\n" + "\n".join(table_list)
        else:
            table_str = "N/A"

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant designed to provide accurate answers based on the official Kolon Benit IT Distribution Business White Paper. 
        Please respond in Korean. 
        Utilize the provided context to answer the question concisely and clearly. 
        If the answer is not found in the context, say that you do not know, and suggest where to find relevant information if possible.
        Ensure your response is structured logically and easy to understand.
        Let's think step-by-step.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question: {query}
        Context: {context_str}
        {image_str}
        {table_str}
        Answer: 
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        logger.debug(f"프롬프트: {prompt}")

    except Exception as search_err:
        logger.error(f"Milvus 검색 호출 실패: {search_err}")

    # watsonx.ai API 요청을 위한 payload 구성
    model_id = request_data.get("model", "meta-llama/llama-3-1-8b-instruct")
    max_tokens = request_data.get("max_tokens", 16392)
    min_tokens = request_data.get("min_tokens", 0)
    repetition_penalty = request_data.get("repetition_penalty", 1)
    stop_sequences = request_data.get("stop_sequences", [])

    # watsonx.ai foundation models
    # payload = {
    #     "input": prompt,
    #     "parameters": {
    #         "decoding_method": "greedy",
    #         "max_new_tokens": max_tokens,
    #         "min_new_tokens": min_tokens,
    #         "stop_sequences": stop_sequences,
    #         "repetition_penalty": repetition_penalty
    #     },
    #     "model_id": model_id,
    #     "project_id": PROJECT_ID
    # }

    # watsonx.ai custom models
    payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty
        }
    }

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        logger.debug(f"watsonx.ai API 요청 payload: {json.dumps(payload, indent=4)}")
        response = requests.post(SOFTWARE_URL, headers=headers, json=payload)
        response.raise_for_status()
        watsonx_response = response.json()
        logger.debug(f"watsonx.ai 응답: {json.dumps(watsonx_response, indent=4)}")
    except Exception as err:
        logger.error(f"watsonx.ai API 호출 에러: {err}")
        raise HTTPException(status_code=500, detail=f"watsonx.ai API 호출 중 에러: {err}")

    # watsonx.ai 응답 처리
    results = watsonx_response.get("results", [])
    if results and "generated_text" in results[0]:
        generated_text = results[0]["generated_text"]
        stop_reason = results[0].get("stop_reason", "stop")
        input_token_count = results[0].get("input_token_count", 0)
        generated_token_count = results[0].get("generated_token_count", 0)
    else:
        logger.error("watsonx.ai 응답에 generated_text가 없습니다.")
        raise HTTPException(status_code=500, detail="watsonx.ai 응답 형식 오류.")

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
            "total_tokens": input_token_count + generated_token_count
        }
    }
