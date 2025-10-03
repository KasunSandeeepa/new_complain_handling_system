"""
Simple prompts for Ollama Llama 3.2 1B LLM integration - Paragraph Format (No Specific Locations)
"""
import re

def create_complaint_solution_prompt(complaint_details: dict, similar_cases: list | None = None, location_info: dict | None = None):
    """
    Create a comprehensive prompt for generating complaint solutions in paragraph format without specific locations
    """
    
    prompt = f"""You are a Sri Lankan telecom expert AI assistant. Generate prioritized technical solutions for this customer complaint:

COMPLAINT DETAILS:
- Issue Description: {complaint_details.get('complaint', 'N/A')}
- Device/Settings: {complaint_details.get('device_type_settings_vpn_apn', 'N/A')}
- Signal Strength: {complaint_details.get('signal_strength', 'N/A')}
- Quality of Signal: {complaint_details.get('quality_of_signal', 'N/A')}
- Site Status: {complaint_details.get('site_kpi_alarm', 'N/A')}
- Past Data: {complaint_details.get('past_data_analysis', 'N/A')}
- Coverage Type: {complaint_details.get('indoor_outdoor_coverage_issue', 'N/A')}
- General Area: {complaint_details.get('location', 'N/A')}

"""

    if similar_cases:
        prompt += "\nSIMILAR PAST CASES:\n"
        for i, case in enumerate(similar_cases[:3], 1):
            issue_desc = case.get('Issue Description', 'N/A')
            solution = case.get('Solution', 'N/A')
            # Remove specific location references from historical cases
            issue_desc = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'nearby site', issue_desc)
            solution = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'the area', solution)
            prompt += f"{i}. Issue: {issue_desc}\n"
            prompt += f"   Solution: {solution}\n"
            prompt += f"   Signal: {case.get('Signal Strength', 'N/A')}\n\n"
    
    if location_info:
        prompt += f"\nAREA NETWORK DATA:\n"
        prompt += f"- Coverage Quality: {location_info.get('coverage_quality', 'N/A')}\n"
        prompt += f"- Signal Distribution: {location_info.get('signal_distribution', 'N/A')}\n"
    
    prompt += """
SOLUTION REQUIREMENTS:
Generate solutions in this EXACT paragraph format:

1. PRIMARY SOLUTION: [Start with most relevant solution] - Provide a small paragraph (2-3 sentences) explaining why this is the most likely solution based on the complaint details and signal conditions. Include specific technical reasoning and expected outcomes.

2. ALTERNATIVE SOLUTION 1: [Second priority solution] - Write a small paragraph explaining this alternative approach, why it might work if the primary solution fails, and what specific conditions make it relevant.

3. ALTERNATIVE SOLUTION 2: [Third priority solution] - Provide a brief paragraph describing this option, including when to consider it and what technical factors support this approach.

4. ADDITIONAL RECOMMENDATION: [Final step] - Write a short paragraph suggesting verification steps, monitoring, or when to escalate to technical support.

IMPORTANT FORMATTING RULES:
- DO NOT mention specific site names, tower codes, or location identifiers
- DO NOT include numerical values like exact signal measurements or coordinates
- Use general terms like "your area", "nearby coverage", "local network" instead of specific locations
- Focus on technical solutions without geographic specifics
- Each solution must be a small paragraph (2-3 meaningful sentences) with clear explanation
- Connect technical reasoning to specific complaint details and conditions
- Explain why each solution is prioritized in that order
- Include practical implementation steps in the explanations
- Use clear, customer-friendly language with technical accuracy
- Focus on Sri Lankan telecom network context and common issues

Generate only the numbered solution paragraphs without additional headings or text:"""

    return prompt

def create_pattern_analysis_prompt(complaint_text: str, historical_data: list):
    """
    Create prompt for analyzing patterns in similar complaints - Paragraph Format (No Specific Locations)
    """
    
    # Clean complaint text of specific locations
    clean_complaint = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'the area', complaint_text)
    
    prompt = f"""Analyze complaint patterns and generate explained solutions:

CURRENT COMPLAINT: {clean_complaint}

HISTORICAL PATTERNS:
"""
    
    for i, case in enumerate(historical_data[:5], 1):
        issue_desc = case.get('Issue Description', 'N/A')
        solution = case.get('Solution', 'N/A')
        # Remove specific location references
        issue_desc = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'nearby site', issue_desc)
        solution = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'the area', solution)
        
        prompt += f"{i}. Issue: {issue_desc}\n"
        prompt += f"   Solution: {solution}\n"
        prompt += f"   Conditions: Signal={case.get('Signal Strength', 'N/A')}\n\n"
    
    prompt += """
Generate solutions in this EXACT paragraph format:

1. PATTERN-BASED PRIMARY SOLUTION: [Most effective historical solution] - Write a small paragraph explaining why this solution has worked for similar cases, referencing general patterns from historical data and how they match the current complaint.

2. PATTERN ALTERNATIVE: [Secondary approach from cases] - Provide a paragraph describing an alternative solution that worked in related scenarios, explaining the conditions where it was successful and why it's relevant here.

3. ADAPTED SOLUTION: [Modified historical approach] - Write a brief paragraph suggesting how to adapt successful historical solutions to the current specific conditions, mentioning any adjustments needed.

4. VERIFICATION & MONITORING: [Pattern validation] - Provide a short paragraph explaining how to verify the solution effectiveness and what patterns to monitor based on historical success metrics.

IMPORTANT: Do not mention specific site names, location codes, or numerical values. Use general terms like "your location", "the network area", or "coverage zone". Generate only the numbered solution paragraphs:"""
    
    return prompt

def create_new_complaint_prompt(complaint_details: dict, location_context: dict | None = None):
    """
    Create prompt for new complaints in paragraph format without specific locations
    """
    
    # Clean complaint of specific locations
    clean_complaint = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'your area', complaint_details.get('complaint', 'N/A'))
    
    prompt = f"""You are analyzing a new telecom complaint in Sri Lanka. Generate explained technical solutions:

NEW COMPLAINT ANALYSIS:
- Issue: {clean_complaint}
- Device/Settings: {complaint_details.get('device_type_settings_vpn_apn', 'N/A')}
- Signal Strength: {complaint_details.get('signal_strength', 'N/A')}
- Quality of Signal: {complaint_details.get('quality_of_signal', 'N/A')}
- Site Status: {complaint_details.get('site_kpi_alarm', 'N/A')}
- Past Data: {complaint_details.get('past_data_analysis', 'N/A')}
- Coverage Type: {complaint_details.get('indoor_outdoor_coverage_issue', 'N/A')}
- General Area: {complaint_details.get('location', 'N/A')}
"""

    if location_context:
        prompt += f"\nAREA CONTEXT:\n"
        prompt += f"- Network Coverage: {location_context.get('coverage_quality', 'Unknown')}\n"
        prompt += f"- Signal Quality: {location_context.get('signal_distribution', 'Unknown')}\n"

    prompt += """
Generate solutions in this EXACT paragraph format:

1. SYMPTOM-BASED PRIMARY SOLUTION: [Most likely approach] - Write a small paragraph analyzing the technical symptoms and explaining why this solution addresses the root cause most effectively. Include specific technical reasoning based on the complaint details.

2. TECHNICAL ALTERNATIVE: [Secondary technical approach] - Provide a paragraph explaining an alternative technical solution, describing different potential root causes and why this approach might be necessary.

3. COMPREHENSIVE TROUBLESHOOTING: [Third approach] - Write a brief paragraph suggesting broader troubleshooting steps, explaining when to use this approach and what additional factors it addresses.

4. ESCALATION RECOMMENDATION: [Final guidance] - Provide a short paragraph explaining when and how to escalate, including what information to provide to technical support for further analysis.

CRITICAL: Avoid mentioning any specific site names, location codes, coordinates, or numerical measurements. Use general terms like "your location", "network in this area", or "local coverage". Generate only the numbered solution paragraphs:"""
    
    return prompt