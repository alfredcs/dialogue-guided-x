from mmrag_gemini import *
if __name__ == '__main__':
   query = "list all companies by names whose Growth Margin rates exceed 80%"
   model_id = 'gemini-pro-vision'
   max_token = 512
   temperature = 0.5
   images_path = "../../langchain/cookbook/cj/"
   mmrag_gemini(query, "../../langchain/cookbook/cj/cj.pdf", images_path, model_id, max_token, temperature) 
