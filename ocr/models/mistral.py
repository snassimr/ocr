from mistralai import Mistral
import time

from .base_model import BaseModel

class mistral(BaseModel):

    def __init__(self,  model_name: str, api_key: str):
         
        super().__init__(model_name, api_key)
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name

        
    def describe(self, frame_urls, prompt):

       
        try:
            
            start_time = time.time()

            response = self.client.ocr.process(
                model = self.model_name,
                document={
                    "type": "image_url",
                    "image_url": frame_urls[0]
                }
            )
            
            end_time = time.time()

            out_text = response.pages[0].markdown.strip()

            processing_time = end_time - start_time

            time.sleep(2)

            return processing_time, out_text
        
        except Exception as e:
            print(f"Error in Mistral VLM: {e}")
            return None
        


