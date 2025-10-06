import time

import google.generativeai as genai


def create_gemini_client(model_name="gemini-2.5-flash"):
    api_keys = [
        "AIzaSyDhrJoRR1ljtgtaRPmk8fcOBJH0WwJh2hU",
        "AIzaSyBqiYdV3DJYCsYaZxqk3mbaajKtUfi5i3s",
        "AIzaSyA6mqtDrwVdJePrXqFOtj8ra-VIHQeXLP0",
        "AIzaSyDf4MjQJycKpxTXUtJRXr4TlJWrYmwNQAM",
        "AIzaSyDdelI3OG34xUPyROt0Q4KWvNl7LyKMtrI",
    ]
    current_idx = 0
    max_retry = len(api_keys)

    # Cấu hình lần đầu
    genai.configure(api_key=api_keys[current_idx])
    model = genai.GenerativeModel(model_name)

    def generate(prompt: str):
        nonlocal current_idx, model
        retry_count = 0

        while retry_count < max_retry:
            try:
                response = model.generate_content(prompt)
                return response.text  # Trả về text
            except Exception as e:
                print(e)
                current_idx = (current_idx + 1) % len(api_keys)
                genai.configure(api_key=api_keys[current_idx])
                model = genai.GenerativeModel(model_name)
                retry_count += 1
                if retry_count >= max_retry:
                    retry_count = 0
                time.sleep(1)  # nghỉ tránh spam

    return generate
