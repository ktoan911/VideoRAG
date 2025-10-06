import google.generativeai as genai

genai.configure(api_key="AIzaSyDhrJoRR1ljtgtaRPmk8fcOBJH0WwJh2hU")

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Xin chào, Gemini!")
print(response.text)
