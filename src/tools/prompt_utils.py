from src.tools.prompt import rewrite, api, get_caption_language

def polish_edit_prompt(instruction, image=None):
    """
    Polishes an edit instruction for the Qwen-Image-Edit pipeline.
    It takes an instruction (and optionally an image) and returns a more
    detailed, model-friendly edit instruction.
    """
    import os
    if "DASH_API_KEY" not in os.environ and "DASHSCOPE_API_KEY" not in os.environ:
        print("Warning: DASH_API_KEY not set. Skipping prompt expansion.")
        return instruction

    lang = get_caption_language(instruction)
    
    if lang == 'hi':
        SYSTEM_PROMPT = '''आप एक चित्र संपादन विशेषज्ञ (Image Editing Expert) हैं। आपका काम उपयोगकर्ता के छोटे संपादन निर्देश को एक विस्तृत और स्पष्ट निर्देश में बदलना है ताकि AI चित्र संपादक उसे अच्छे से समझ सके।
उदाहरण: "कुत्ता जोड़ें" -> "चित्र में एक प्यारा सा सुनहरा कुत्ता जोड़ें, जो घास पर बैठा हो।"
उदाहरण: "इसे रात बनाएं" -> "दृश्य को रात के समय में बदलें, जिसमें तारों भरा आसमान और चाँद की रोशनी हो।"
केवल एक विस्तृत संपादन निर्देश दें, कोई अन्य स्पष्टीकरण न दें।'''
        full_prompt = f"{SYSTEM_PROMPT}\n\nउपयोगकर्ता निर्देश: {instruction}\n\nविस्तृत निर्देश:"
    else:
        SYSTEM_PROMPT = '''You are an expert in image editing instructions. Your task is to take a brief user instruction and expand it into a highly descriptive, clear semantic instruction for an image editing AI.
Focus on adding necessary details for Add, Replace, or Style Change if they are missing.
Example: "add a dog" -> "Add a cute golden retriever dog sitting on the grass."
Example: "make it winter" -> "Change the scene to winter, adding snow on the ground and a cold, frosty atmosphere."
Return ONLY the expanded instruction, without any conversational text or quotes.'''
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Instruction: {instruction}\n\nExpanded Instruction:"
    
    success = False
    max_retries = 3
    retries = 0
    while not success and retries < max_retries:
        try:
            polished = api(full_prompt, model='qwen-plus')
            polished = polished.strip().replace("\n", " ").strip('"\'')
            success = True
        except Exception as e:
            print(f"Error during API call for polish_edit_prompt: {e}")
            retries += 1
            if retries == max_retries:
                return instruction # Fallback to original
            
    return polished
