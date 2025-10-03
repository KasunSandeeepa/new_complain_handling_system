"""
Simplified complaint handling solution system - Paragraph Format (No Specific Locations)
"""

import pandas as pd
import pickle
import hashlib
import os
import requests
import re
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory to path to import prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------- Fallback prompts --------------------
try:
    from prompts import create_complaint_solution_prompt, create_pattern_analysis_prompt, create_new_complaint_prompt
except ImportError:
    print("Warning: Could not import prompts module. Using fallback prompts.")
    
    def create_complaint_solution_prompt(complaint_details, similar_cases=None, location_info=None):
        return f"Generate paragraph solutions for: {complaint_details.get('complaint', 'N/A')}"
    
    def create_pattern_analysis_prompt(complaint_text, historical_data):
        return f"Analyze patterns for: {complaint_text} in paragraph format"
    
    def create_new_complaint_prompt(complaint_details, location_context=None):
        return f"Generate new paragraph solutions for: {complaint_details.get('complaint', 'N/A')}"

# -------------------- Helper functions --------------------
def remove_specific_locations(text: str) -> str:
    if not text or not text.strip():
        return text
    text = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'the area', text)
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b\d+dBm\b', 'current signal level', text)
    text = re.sub(r'\b\d+%\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s,', ',', text)
    return text.strip()

def format_solution_paragraphs(solution_text: str, solution_type: str) -> str:
    if not solution_text or not solution_text.strip():
        return "No solution generated."
    cleaned_text = remove_specific_locations(solution_text.strip())
    paragraphs = []
    raw_paragraphs = re.split(r'\n\s*\d+\.', cleaned_text)
    raw_paragraphs = [para.strip() for para in raw_paragraphs if para.strip()]
    if len(raw_paragraphs) >= 2:
        for i, paragraph in enumerate(raw_paragraphs[:4], 1):
            clean_para = remove_specific_locations(paragraph)
            clean_para = re.sub(r'^\s*[A-Z ]+:\s*', '', clean_para)
            paragraphs.append(f"{i}. {clean_para.strip()}")
    else:
        raw_paragraphs = re.split(r'\n\s*\n', cleaned_text)
        raw_paragraphs = [para.strip() for para in raw_paragraphs if para.strip()]
        for i, paragraph in enumerate(raw_paragraphs[:4], 1):
            clean_para = remove_specific_locations(paragraph)
            paragraphs.append(f"{i}. {clean_para}")
    if not paragraphs:
        paragraphs = ["1. Please contact technical support for detailed analysis and assistance with this issue."]
    formatted_solution = "\n\n".join(paragraphs)
    formatted_solution += f"\n\n[Solution type: {solution_type}]"
    return formatted_solution

# -------------------- New LLM output formatter --------------------
def format_llm_solution(raw_text: str) -> str:
    """
    Format raw LLM solution into a clean, formal paragraph format with headings.
    """
    if not raw_text or not raw_text.strip():
        return "No solution generated."
    text = re.sub(r'\.\s*\.', '.', raw_text)
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'(PRIMARY SOLUTION|ALTERNATIVE SOLUTION|ADDITIONAL RECOMMENDATION)', text, flags=re.IGNORECASE)
    formatted_parts = []
    current_number = 1
    i = 1
    while i < len(parts):
        header = parts[i].strip().upper()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        if "PRIMARY" in header:
            formatted_parts.append(f"{current_number}. Primary Solution:\n{content}\n")
        elif "ALTERNATIVE" in header:
            current_number += 1
            formatted_parts.append(f"{current_number}. Alternative Solution:\n{content}\n")
        elif "ADDITIONAL" in header:
            current_number += 1
            formatted_parts.append(f"{current_number}. Additional Recommendation:\n{content}\n")
        i += 2
    return "\n".join(formatted_parts).strip()

# -------------------- Main Complaint Handler --------------------
class SimpleComplaintHandler:
    def __init__(self, model_path: str = "models/unified_complaint_model.pkl"):
        self.model_path = model_path
        self.complaint_data = None
        self.location_data = None
        self.ollama_url = "http://localhost:11434"
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.complaint_data = model_data.get('complaint_data')
                self.location_data = model_data.get('location_data')
                print(f"Loaded unified model with {len(self.complaint_data)} complaints and {len(self.location_data)} locations")
            else:
                print(f"Model file {self.model_path} not found. Please train the model first.")
                self.complaint_data = pd.DataFrame()
                self.location_data = pd.DataFrame()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.complaint_data = pd.DataFrame()
            self.location_data = pd.DataFrame()

    # -------------------- Signature & Exact Match --------------------
    def create_complaint_signature(self, complaint_text: str, conditions: Dict | None = None) -> str:
        normalized_complaint = remove_specific_locations(complaint_text.lower().strip())
        condition_str = ""
        if conditions:
            all_conditions = [
                'device_type_settings_vpn_apn', 'signal_strength', 'quality_of_signal', 
                'site_kpi_alarm', 'past_data_analysis', 'indoor_outdoor_coverage_issue',
                'location', 'longitude', 'latitude'
            ]
            condition_values = []
            for key in all_conditions:
                value = conditions.get(key)
                if value is not None and str(value).strip().lower() not in ['','n/a','none','null','undefined']:
                    if key in ['longitude','latitude'] and isinstance(value,(int,float)):
                        if abs(float(value))>0.001:
                            condition_values.append(f"{key}:{round(float(value),6)}")
                    else:
                        condition_values.append(f"{key}:{str(value).lower().strip()}")
            condition_str = "|".join(sorted(condition_values))
        signature_input = f"{normalized_complaint}|{condition_str}"
        return hashlib.md5(signature_input.encode()).hexdigest()

    def find_exact_match(self, complaint_text: str, conditions: Dict | None = None) -> Optional[str]:
        if self.complaint_data is None or self.complaint_data.empty:
            return None
        current_signature = self.create_complaint_signature(complaint_text, conditions)
        for _, row in self.complaint_data.iterrows():
            hist_conditions = {}
            for col,key in [('Device type/settings/VPN/APN','device_type_settings_vpn_apn'),
                            ('Signal Strength','signal_strength'),
                            ('Qulity of Signal','quality_of_signal'),
                            ('Site KPI/Alarm','site_kpi_alarm'),
                            ('Past Data analysis','past_data_analysis'),
                            ('Indoor/Outdoor coverage issue','indoor_outdoor_coverage_issue')]:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    hist_conditions[key] = str(val)
            for coord_col, coord_key in [('Lon','longitude'), ('Lat','latitude')]:
                try:
                    coord_val = row.get(coord_col)
                    if pd.notna(coord_val):
                        fval = float(coord_val)
                        if abs(fval)>0.001:
                            hist_conditions[coord_key] = fval
                except:
                    continue
            hist_issue_desc = remove_specific_locations(str(row.get('Issue Description','')))
            hist_signature = self.create_complaint_signature(hist_issue_desc,hist_conditions)
            if current_signature==hist_signature:
                exact_solution = str(row.get('Solution','No solution found'))
                return format_solution_paragraphs(exact_solution,"exact_match")
        return None

    # -------------------- Location context --------------------
    def get_location_context(self, complaint_details: Dict) -> Optional[Dict]:
        if self.location_data is None or self.location_data.empty:
            return None
        longitude = complaint_details.get('longitude')
        latitude = complaint_details.get('latitude')
        location_name = complaint_details.get('location')
        location_context = None
        if longitude is not None and latitude is not None:
            min_distance=float('inf')
            closest_location=None
            for _,row in self.location_data.iterrows():
                try:
                    row_lon=float(row.get('Lon',0))
                    row_lat=float(row.get('Lat',0))
                    distance=((longitude-row_lon)**2+(latitude-row_lat)**2)**0.5
                    if distance<min_distance and distance<0.01:
                        min_distance=distance
                        closest_location=row.to_dict()
                except:
                    continue
            if closest_location:
                location_context={'coverage_quality':'Good' if float(str(closest_location.get('RSRP >-105dBm (%)','0')).replace('%',''))>70 else 'Poor',
                                  'signal_distribution':'Good coverage, Fair coverage, Poor coverage based on area signal quality'}
        if not location_context and location_name:
            for _,row in self.location_data.iterrows():
                site_name=str(row.get('Site Name','')).lower()
                if location_name.lower() in site_name or site_name in location_name.lower():
                    location_context={'coverage_quality':'Good' if float(str(row.get('RSRP >-105dBm (%)','0')).replace('%',''))>70 else 'Poor',
                                      'signal_distribution':'Good coverage, Fair coverage, Poor coverage based on area signal quality'}
                    break
        return location_context

    # -------------------- Similar complaints --------------------
    def find_similar_complaints(self, complaint_text: str, top_n: int = 5) -> List[Dict]:
        if self.complaint_data is None or self.complaint_data.empty:
            return []
        clean_complaint=remove_specific_locations(complaint_text.lower())
        complaint_words=set(clean_complaint.split())
        similarities=[]
        for _,row in self.complaint_data.iterrows():
            issue_desc=remove_specific_locations(str(row.get('Issue Description','')))
            issue_words=set(issue_desc.lower().split())
            intersection=len(complaint_words & issue_words)
            union=len(complaint_words | issue_words)
            similarity=intersection/union if union>0 else 0
            if similarity>0.1:
                row_data=row.to_dict()
                if 'Solution' in row_data:
                    row_data['Solution']=remove_specific_locations(str(row_data['Solution']))
                if 'Issue Description' in row_data:
                    row_data['Issue Description']=remove_specific_locations(str(row_data['Issue Description']))
                similarities.append({'similarity':similarity,'data':row_data})
        similarities.sort(key=lambda x:x['similarity'],reverse=True)
        return [item['data'] for item in similarities[:top_n]]

    # -------------------- Call Ollama LLM --------------------
    def call_ollama_llm(self,prompt:str,model:str="llama3:latest")->str:
        try:
            response=requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model":model,"prompt":prompt,"stream":False,"options":{"temperature":0.7,"max_tokens":350,"top_p":0.9}},
                timeout=1500
            )
            if response.status_code==200:
                result=response.json()
                response_text=result.get('response','Unable to generate solution')
                return format_llm_solution(response_text)
            else:
                return f"LLM Error: HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "Error: Unable to connect to Ollama. Please ensure Ollama is running with the correct model."
        except requests.exceptions.ReadTimeout:
            return "Error calling LLM: Read timed out."
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    # -------------------- Main generate solution --------------------
    def generate_solution(self, complaint_details: Dict) -> Tuple[str,str]:
        complaint_text=complaint_details.get('complaint','')
        exact_solution=self.find_exact_match(complaint_text,complaint_details)
        if exact_solution:
            return exact_solution,"exact_match"
        location_context=self.get_location_context(complaint_details)
        similar_complaints=self.find_similar_complaints(complaint_text)
        if similar_complaints:
            if location_context and len(similar_complaints)>=3:
                prompt=create_complaint_solution_prompt(complaint_details,similar_complaints,location_context)
                solution=self.call_ollama_llm(prompt)
                return solution,"comprehensive_analysis"
            else:
                prompt=create_pattern_analysis_prompt(complaint_text,similar_complaints)
                solution=self.call_ollama_llm(prompt)
                return solution,"pattern_analysis"
        prompt=create_new_complaint_prompt(complaint_details,location_context)
        solution=self.call_ollama_llm(prompt)
        return solution,"new_complaint"

# -------------------- Global instance --------------------
complaint_handler=None
def get_complaint_handler():
    global complaint_handler
    if complaint_handler is None:
        complaint_handler=SimpleComplaintHandler()
    return complaint_handler

def generate_solution(msisdn:str,complaint_text:str,**kwargs)->str:
    handler=get_complaint_handler()
    complaint_details={'msisdn':msisdn,
                       'complaint':complaint_text,
                       'device_type_settings_vpn_apn':kwargs.get('device_type_settings_vpn_apn'),
                       'signal_strength':kwargs.get('signal_strength'),
                       'quality_of_signal':kwargs.get('quality_of_signal'),
                       'site_kpi_alarm':kwargs.get('site_kpi_alarm'),
                       'past_data_analysis':kwargs.get('past_data_analysis'),
                       'indoor_outdoor_coverage_issue':kwargs.get('indoor_outdoor_coverage_issue'),
                       'location':kwargs.get('location'),
                       'longitude':kwargs.get('longitude'),
                       'latitude':kwargs.get('latitude')}
    solution,solution_type=handler.generate_solution(complaint_details)
    return solution
