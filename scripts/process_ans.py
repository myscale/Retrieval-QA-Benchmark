import re

def extract_and_format(text):
    # 更新后的正则表达式模式
    pattern = r"Throughput: (\d+\.\d+|\d+) req/s, (\d+) prompt tokens/s \(avg length (\d+)\), (\d+) completion tokens/s \(avg_length (\d+)\)"
    match = re.search(pattern, text)

    if match:
        req_per_s = match.group(1)
        prompt_tokens = match.group(2)
        avg_length_prompt = match.group(3)
        completion_tokens = match.group(4)
        avg_length_completion = match.group(5)
        
        return f"{req_per_s} req/s | {prompt_tokens}/{avg_length_prompt} | {completion_tokens}/{avg_length_completion}"
    else:
        return None

def process_multiline_text(multiline_text):
    lines = multiline_text.split('\n')
    result = []

    for line in lines:
        formatted_line = extract_and_format(line)
        if formatted_line:
            result.append(formatted_line)

    return '\n'.join(result)

# 从文件读取文本
with open("ans_batch1.txt", "r") as file:
    multiline_text = file.read()

print(process_multiline_text(multiline_text))
