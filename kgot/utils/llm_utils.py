# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Main author: Lorenzo Paleari

import json
import logging
import traceback

import httpx
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, InternalServerError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    wait_random_exponential,
)

CONFIG_LLM_PATH = ''
NUM_LLM_RETRIES = 1

logger = logging.getLogger("Controller.LLMUtils")

def init_llm_utils(config_path: str = CONFIG_LLM_PATH, 
                   num_retries: int = NUM_LLM_RETRIES):
    """
    Initialize the LLM utils with the given configuration path.
    """
    global CONFIG_LLM_PATH
    CONFIG_LLM_PATH = config_path
    global NUM_LLM_RETRIES
    NUM_LLM_RETRIES = num_retries
    logger.info(f"LLM utils initialized with config path: {CONFIG_LLM_PATH} and num retries: {NUM_LLM_RETRIES}")


def _get_llm_retries():
    """
    Get the number of retries for LLM requests.
    """
    global NUM_LLM_RETRIES
    return NUM_LLM_RETRIES


def _uses_openai_responses_api(chain: ChatOpenAI) -> bool:
    if isinstance(getattr(chain, "use_responses_api", None), bool):
        return chain.use_responses_api
    return any(
        getattr(chain, attr, None) is not None
        for attr in ["reasoning", "include", "truncation"]
    ) or bool(getattr(chain, "use_previous_response_id", False))


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is not None:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content)


def _truncate_response_at_stop(response, stop):
    if isinstance(stop, str):
        stop = [stop]
    if (
        not stop
        or not hasattr(response, "content")
    ):
        return response
    content = _content_to_text(response.content)
    stop_positions = [content.find(seq) for seq in stop if seq]
    stop_positions = [position for position in stop_positions if position >= 0]
    if stop_positions:
        content = content[:min(stop_positions)]
    response.content = content
    return response


def invoke_with_retry(chain: (ChatOpenAI | ChatOllama), *args, **kwargs):
    stop = kwargs.pop('stop', None)
    omit_stop = (
        stop is not None
        and isinstance(chain, ChatOpenAI)
        and _uses_openai_responses_api(chain)
    )
    try: 
        for attempt in Retrying(
            wait=wait_random_exponential(min=1, max=60), 
            stop=_get_llm_retries(), 
            reraise=True,
            retry=(
                retry_if_exception_type(InternalServerError) |
                retry_if_exception_type(APIConnectionError) |
                retry_if_exception_type(httpx.ConnectTimeout) |
                retry_if_exception_type(httpx.ReadTimeout)
            )
        ):
            with attempt:
                try:
                    if omit_stop:
                        return _truncate_response_at_stop(
                            chain.invoke(*args, **kwargs),
                            stop,
                        )
                    if stop is not None:
                        return chain.invoke(*args, stop=stop, **kwargs)
                    return chain.invoke(*args, **kwargs)
                except InternalServerError as e:
                    logger.error(f"Internal Server Error when invoking the chain: {str(e)} - Type of error: {type(e)}")
                    raise  # Re-raise the exception to trigger retry
                except APIConnectionError as e:
                    logger.error(f"API Connection Error when invoking the chain: {str(e)} - Type of error: {type(e)}")
                    raise
                except Exception as e:
                    logger.error(f"Error when invoking the chain: {str(e)} - Type of error: {traceback.format_exc()}")
                    raise
    except Exception as e:
        logger.error(f"Failed to invoke the chain after {NUM_LLM_RETRIES} attempts: {str(e)}")
        raise


def get_model_configurations(model_name: str) -> dict:
    global CONFIG_LLM_PATH
    with open(CONFIG_LLM_PATH, 'r') as config_file:
        config = json.load(config_file)

    if model_name is None:
        raise ValueError(f"Model '{model_name}' not found in the supported models. Update the config_llms.json file")

    return config[model_name]


def get_llm(model_name: str, temperature: float = None, max_tokens: int = None):
    # Set up the LLMs
    model_config = get_model_configurations(model_name)
    if temperature is not None:
        model_config["temperature"] = temperature
    if not (0 <= model_config["temperature"] <= 1):
        raise ValueError(
            f"LLM models temperature needs to be in the range [0, 1], but given {model_config['temperature']}")
    model_config["max_tokens"] = max_tokens

    llm_to_return = None
    if model_config["model_family"] == "OpenAI":
        llm_to_return = ChatOpenAI(
            model=model_config["model"],
            api_key=model_config["api_key"],
            max_tokens=model_config["max_tokens"],
            organization=model_config["organization"],
            **{key: model_config[key] for key in 
               ["temperature", "reasoning"] if key in model_config}
        )
    elif model_config["model_family"] == "Ollama":
        llm_to_return = ChatOllama(
            model=model_config["model"],
            temperature=model_config["temperature"],
            base_url=model_config["base_url"] if "base_url" in model_config else "localhost:11434",
            num_ctx=model_config["num_ctx"],
            num_predict=model_config["num_predict"],
            num_batch=model_config["num_batch"],
            keep_alive=-1)
    else:
        raise ValueError(f"Model family '{model_config['model_family']}' not supported, check the config_llms.json file")

    return llm_to_return
