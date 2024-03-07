import re
from pathlib import Path
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
from openai import OpenAI
from template import DAT_CLS_TEMPLATE1, DAT_CLS_TEMPLATE2, postprocess_ie


def load_qwen_model(model_dir, device, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True, **kwargs).eval()
    return model, tokenizer


def qwen_infer(model, tokenizer, query, temperature=0.51, top_p=0.7):

    query_imgf = query[0]['image']
    query_text = query[-1]['text']

    query = tokenizer.from_list_format(query)
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs, temperature=temperature, top_p=top_p)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

    pattern = rf"{query_text}(.*?)<|endoftext|>$"
    matches = re.search(pattern, response, re.DOTALL)
    if matches:
        results = matches.group(1).strip()
        if len(results) > 0:
            txt_f = Path(query_imgf).with_suffix(".txt")
            with open(txt_f, "w") as f:
                f.write(results)


def qwen_chat_infer(model, tokenizer, query, save_root, openai_client, history=None, save_json=False, **kwargs):
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)

    query_imgf = query[0]['image']
    # query_text = query[-1]['text']
    query = tokenizer.from_list_format(query)
    response, history = model.chat(tokenizer, query=query, history=history, **kwargs)
    txt_f = save_root / (Path(query_imgf).stem + ".txt")
    with open(txt_f, "w") as f:
        f.write(response)

    if save_json:
        try:
            feat_dict = parser_response_to_feat_dict(openai_client, response)
            json_f = txt_f.with_suffix(".json")
            with open(json_f, "w") as f:
                json.dump(feat_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            pass
    return response, history



def parser_response_to_feat_dict(openai_client, response):
    chat_response = openai_client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[
            # {"role": "system", "content": DAT_CLS_TEMPLATE3},
            {"role": "user", "content": f"{DAT_CLS_TEMPLATE1}{response}{DAT_CLS_TEMPLATE2}"},
        ],
        stream=False,
        stop=[],
    )
    msg = chat_response.choices[0].message.content.strip()
    feat_dict = postprocess_ie(msg)
    return feat_dict


def get_feat_dict_from_txt(root, openai_client):
    for txt_f in root.iterdir():
        if txt_f.suffix == ".txt":
            with open(txt_f, "r") as f:
                # lines = f.readlines()
                # content = lines[-1]
                content = f.read()
            feat_dict = parser_response_to_feat_dict(openai_client, content)
            json_f = txt_f.with_suffix(".json")
            with open(json_f, "w") as f:
                json.dump(feat_dict, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    root = Path("human_struct_test_data/test")
    save_root = root 

    model, tokenizer = load_qwen_model("Qwen-VL-Chat", "cuda")
    
                    

    for jpg_f in root.iterdir():
        if jpg_f.suffix == ".jpg":
            query = [
                {'image': str(jpg_f)},
                {'text': '请描述图中这个人的性别，衣服，发式，鞋等外观。（比如是否戴眼镜，是否佩戴口罩，是否携带包）'},
            ]
            res = qwen_chat_infer(model, tokenizer, query, root, client)
