import os
import re
import json

def read_txt_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def extract_fields(input_string):
    pattern = r'(\w+): ([^\n]+)'
    matches = re.findall(pattern, input_string)
    result = {key: value.strip() for key, value in matches}
    return result

def decode_response(response):
    """
    Get the formatted answer from MLLM's response
    """
    for ch in ['"', "'", '{', '}']:
        response = response.replace(ch, '')
    resp_json = extract_fields(response)
    response_json = {k.lower(): v for k, v in resp_json.items()}
    if 'description' in response_json:
        response_json['Image description'] = response_json.pop('description')
    if 'image_description' in response_json:
        response_json['Image description'] = response_json.pop('image_description')
    if 'imagedescription' in response_json:
        response_json['Image description'] = response_json.pop('imagedescription')
    if 'reasoning' in response_json:
        response_json['Forgery reasoning'] = response_json.pop('reasoning')
    if 'forgery_reasoning' in response_json:
        response_json['Forgery reasoning'] = response_json.pop('forgery_reasoning')
    if 'forgeryreasoning' in response_json:
        response_json['Forgery reasoning'] = response_json.pop('forgeryreasoning')
    if 'result' in response_json:
        response_json['Analysis result'] = response_json.pop('result')
    if 'analysis_result' in response_json:
        response_json['Analysis result'] = response_json.pop('analysis_result')
    if 'analysisresult' in response_json:
        response_json['Analysis result'] = response_json.pop('analysisresult')
    if 'type' in response_json:
        response_json['Forgery type'] = response_json.pop('type')
    if 'forgery_type' in response_json:
        response_json['Forgery type'] = response_json.pop('forgery_type')
    if 'forgerytype' in response_json:
        response_json['Forgery type'] = response_json.pop('forgerytype')
    if 'probability' in response_json:
        response_json['Probability'] = response_json.pop('probability')

    if not 'reasoning' in response_json and 'analysis' in response_json: #zzzzzz
        response_json['Forgery reasoning'] = response_json.pop('analysis')
    if 'Forgery type' in response_json and not 'Analysis result' in response_json:
        if response_json['Forgery type'].lower() == 'none':
            response_json['Analysis result'] = 'real'
        else:
            response_json['Analysis result'] = 'fake'
    if not 'Forgery type' in response_json and not 'Analysis result' in response_json:
        print(response_json)
        response_json['Forgery type'] = 'PAD attacks'
        response_json['Analysis result'] = 'fake'

    if response_json['Analysis result'].lower() != 'real' and response_json['Analysis result'].lower() != 'fake':
        response_json['Analysis result'] = 'fake'

    key_order = ['Image description', 'Forgery reasoning', 'Analysis result', 'Probability', 'Forgery type']
    sorted_response_json = {key: response_json[key] for key in key_order if key in response_json}
    formatted_response = "\n".join(f"{key}: {value}" for key, value in sorted_response_json.items())
    return sorted_response_json, formatted_response

def mask_result(answer):
    """
    Mask analysis result of the answer
    """
    answer_json, _ = decode_response(answer)
    answer_result = answer_json['Analysis result'].lower()
    new_answer_json = {
        'Image description': answer_json['Image description'],
        'Forgery reasoning': answer_json['Forgery reasoning']
    }
    new_answer = "\n".join(f"{key}: {value}" for key, value in new_answer_json.items())
    return new_answer, answer_result

def answer_format(answer_json):
    """
    whether output formatted answer 
    """
    del answer_json['Probability']
    data = answer_json

    key_order = ['Image description', 'Forgery reasoning', 'Analysis result', 'Forgery type', 'Match score', 'Difficulty']
    sorted_response_json = {key: data[key] for key in key_order if key in data}
    formatted_string = "\n".join(f"{key}: {value}" for key, value in sorted_response_json.items())

    # formatted_string = (
    #     f"Image description: {data['Image description']}\n"
    #     f"Forgery reasoning: {data['Forgery reasoning']}\n"
    #     f"Analysis result: {data['Analysis result']}, Forgery type: {data['Forgery type']}\n"
    #     f"Match score: {data['Match score']}; Difficulty: {data['Difficulty']}"
    # )
    return formatted_string

def get_result(answer):
    """
    Get the binary classification result from the anwswer
    """
    answer_json, _ = decode_response(answer)
    answer_result = answer_json['Analysis result'].lower()
    return answer, answer_result

# not much use
def decode_outputs(imgname, gt, outputs):
    result = {}
    response_json, formatted_response = decode_response(outputs)
    result['id'] = imgname
    if 'Analysis result' in response_json and 'Forgery type' in response_json:
        result['2-class-correct'] = response_json['Analysis result'] == gt[imgname][0]
        result['4-class-correct'] = response_json['Forgery type'] == gt[imgname][1]
    else:
        result['2-class-correct'] = False
        result['4-class-correct'] = False
    result['content'] = formatted_response
    return result
    
def read_json(json_file):
    with open(json_file, 'r') as json_f:
        image_info_dict = json.load(json_f)
    return image_info_dict

def load_jsonl(json_file):
    data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def write_jsonl(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

def write_json(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def unique_filename_by_size(path, org_file):
    """
    Returns a filename that matches or avoids size conflicts.
    - If 'path' exists and size matches expected_size → return 'path'
    - If size differs → increment index: file_1.ext, file_2.ext, ...
    """
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    # First try the original filename
    candidate = path
    counter = 1
    expected_size = os.path.getsize(org_file)

    while True:
        if os.path.exists(candidate):
            existing_size = os.path.getsize(candidate)

            if existing_size == expected_size:
                # Same size → reuse
                return candidate
            else:
                # Different size → try next index
                candidate = os.path.join(directory, f"{name}_{counter}{ext}")
                counter += 1
        else:
            # File does not exist → safe to use
            return candidate

def get_jsonfmt(text):
    result = {}

    def normalize_key(k):
        return k.strip().lower().strip('"').strip("'")

    def normalize_value(v):
        v = v.strip().strip('"').strip("'")

        # Convert float
        if re.fullmatch(r"\d+\.\d+", v):
            return float(v)

        # Convert int
        if v.isdigit():
            return float(v)

        return v

    # Split by lines
    for line in text.splitlines():
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "Image description":
            items = [item.strip() for item in value.split(",")]
            desc = {}
            last_key = None  # keep track of previous key
            for item in items:
                if "-" in item:
                    k, v = item.split("-", 1)
                    k = normalize_key(k)
                    v = normalize_value(v)
                    desc[k] = v
                    last_key = k
                else:
                    # No "-", append to previous item's value
                    if last_key is not None:
                        desc[last_key] += f", {item}"
                    # else: ignore safely if no previous key exists

            result[key] = desc
        else:
            # Convert numeric values when possible
            if re.fullmatch(r"\d+\.\d+", value):
                value = float(value)
            result[key] = value

    # 🔍 Quality-based override logic
    # quality = float(result.get("Image description", {}).get("quality"))

    # if quality is not None and quality < 0.5 and result["Analysis result"] == "real":
    #     result["Analysis result"] = "likely_fake"
    #     result["Forgery type"] = "Bad Quality"
    #     result["Forgery reasoning"] += (
    #         " However, the image quality is so poor that it is very likely to be fake."
    #     )
    quality = result.get("Image description", {}).get("quality").lower() #zzzzzzzz
    if quality is not None and quality in ["low", "poor"] and result["Analysis result"] == "real":
        result["Analysis result"] = "likely_fake"
        result["Forgery type"] = "Bad Quality"
        result["Forgery reasoning"] += (
            " However, the image quality is so poor that it is very likely to be fake."
        )


    # Convert to JSON
    # json_output = json.dumps(result, indent=2)

    return result

def fix_blured_real(best_answer_json, answers, answers_result):
    # blur_phrases = [
    #     "image is slightly blurry, but this does not suggest forgery",
    #     "the image is slightly blurry, but these do not indicate forgery",
    #     "slight blurriness is noted, but it does not indicate forgery",
    #     "slight blurriness does not indicate forgery",
    #     "slight blurriness is present but does not indicate forgery",
    #     "slight blurriness are present but do not indicate forgery",
    # ]
    BLUR_NOT_FORGERY_PATTERN = re.compile(
        r"(slight(ly)?\s+blur(riness|ry)|image\s+is\s+slight(ly)?\s+blurry)"
        r".*?\b(do|does)\s+not\s+(indicate|suggest)\s+forgery",
        re.IGNORECASE,
    )
    def contains_blur_not_forgery(reasoning: str) -> bool:
        return bool(BLUR_NOT_FORGERY_PATTERN.search(reasoning))

    reasoning = best_answer_json.get("Forgery reasoning", "").lower()

    # If no blur-related "not forgery" phrase exists, do nothing
    # if not any(phrase in reasoning for phrase in blur_phrases):
    #     return best_answer_json
    if not contains_blur_not_forgery(reasoning):
        return best_answer_json

    # Blur is present but should not be treated as real
    best_answer_json["Analysis result"] = "likely_fake"

    if best_answer_json.get("Difficulty") == "hard":
        # Find the last answer labeled as fake
        for idx in range(len(answers_result) - 1, -1, -1):
            if answers_result[idx] == "fake":
                selected_json, _ = decode_response(answers[idx])
                best_answer_json["Forgery type"] = selected_json["Forgery type"]
                best_answer_json["Forgery reasoning"] = (selected_json["Forgery reasoning"] +
                                                         " The photo is blurry, so it's likely fake.")
                break
    else:
        best_answer_json["Forgery reasoning"] += (
            " The photo is blurry, so it's likely fake."
        )

    return best_answer_json
