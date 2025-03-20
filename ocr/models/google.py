import google.generativeai as genai
import time

from .base_model import BaseModel


class Gemini(BaseModel):

    def __init__(self,  model_name: str, api_key: str):
         
        super().__init__(model_name, api_key)
        genai.configure(api_key=api_key)
        self.model_name = model_name
        
        
        
    def describe(self, frame_urls, prompt):
        
        
        images = [self.encode_image(url) for url in frame_urls]
     
       
        model = genai.GenerativeModel(self.model_name)
        
        try:   
            
            image_dicts = [
                {
                    'mime_type': 'image/jpeg',
                    'data': img
                }
                for img in images ]
    
            content_parts = image_dicts + [prompt]
            
            start_time = time.time()
            
            response = model.generate_content(content_parts, stream=True)
           
            response.resolve()
            
            end_time = time.time()

            out_text = self.to_markdown(response.text)
            cleaned_out_text = self.clean_ocr_text(out_text)

            processing_time = end_time - start_time
        
            return processing_time, cleaned_out_text
        
        except Exception as e:
            raise AttributeError(f"Error in Google VLM: {e}")

from google import genai
from google.genai import types
from PIL import Image
import time

class Gemini2(BaseModel):

    def __init__(self,  model_name: str, api_key: str):
         
        super().__init__(model_name, api_key)

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        
    def describe(self, frame_urls, prompt):
                
        images = [self.load_image(path) for path in frame_urls]
        
        content_parts = images + [prompt]
        
        try:   
            
            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT']
                )
            )
            
            end_time = time.time()

            out_text = self.to_markdown(response.text)
            cleaned_out_text = self.clean_ocr_text(out_text)

            processing_time = end_time - start_time

            time.sleep(2)
        
            return processing_time, cleaned_out_text
        
        except Exception as e:
            raise AttributeError(f"Error in Google VLM: {e}")
        


