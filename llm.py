"""
llm.py — LLM integration with Groq for construction compliance analysis
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from ingestion import Chunk


@dataclass
class ComplianceCheck:
    """Structured compliance assessment result"""
    status: str  # "PASS", "FAIL", "WARNING", "UNKNOWN"
    severity: str  # "HIGH", "MEDIUM", "LOW", "N/A"
    rule_value: Optional[str]  # What the regulation requires
    project_value: Optional[str]  # What the project shows
    explanation: str  # Human-readable explanation


@dataclass
class LLMResult:
    """Complete LLM response with answer and optional compliance check"""
    answer: str
    compliance_check: Optional[ComplianceCheck]
    prompt: str  # The full prompt sent to LLM


def is_compliance_question(query: str) -> bool:
    """Check if the query is asking about compliance/conformance"""
    compliance_indicators = [
        "comply", "compliance", "compliant", "conform", "conformance",
        "meet", "meets", "satisfy", "satisfies", "pass", "fail",
        "according to", "as per", "required by", "standard",
        "regulation", "code", "specification", "rule"
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in compliance_indicators)


def _extract_compliance_info(response_text: str) -> Optional[ComplianceCheck]:
    """
    Parse LLM response for structured compliance information.
    Looks for patterns like:
    - Status: PASS/FAIL/WARNING
    - Required: [value]
    - Actual: [value]
    """
    # Try to extract structured compliance info
    status_match = re.search(r'Status:\s*(PASS|FAIL|WARNING|UNKNOWN)', response_text, re.IGNORECASE)
    severity_match = re.search(r'Severity:\s*(HIGH|MEDIUM|LOW|N/A)', response_text, re.IGNORECASE)
    required_match = re.search(r'Required:\s*([^\n]+)', response_text, re.IGNORECASE)
    actual_match = re.search(r'Actual:\s*([^\n]+)', response_text, re.IGNORECASE)
    
    if not status_match:
        return None
    
    # Extract explanation (everything after the structured parts)
    explanation = response_text
    for match in [status_match, severity_match, required_match, actual_match]:
        if match:
            explanation = explanation.replace(match.group(0), "")
    explanation = explanation.strip()
    
    return ComplianceCheck(
        status=status_match.group(1).upper(),
        severity=severity_match.group(1).upper() if severity_match else "N/A",
        rule_value=required_match.group(1).strip() if required_match else None,
        project_value=actual_match.group(1).strip() if actual_match else None,
        explanation=explanation or "No additional explanation provided."
    )


def _build_prompt(query: str, context_chunks: List[Chunk]) -> str:
    """Build the complete prompt for the LLM"""
    
    # Separate codebook and specification chunks
    codebook_chunks = [c for c in context_chunks if c.document_type == "codebook"]
    spec_chunks = [c for c in context_chunks if c.document_type == "specification"]
    
    # Get project info from chunks
    project_ids = set(c.project_id for c in context_chunks if c.project_id)
    project_id = project_ids.pop() if project_ids else "Unknown Project"
    
    prompt = f"""You are a construction compliance expert analyzing building codes and project specifications.

PROJECT CONTEXT:
You are analyzing documents for PROJECT {project_id}. All provided documents relate to the SAME PROJECT.

QUERY: {query}

DOCUMENT CONTEXT:
"""
    
    if codebook_chunks:
        prompt += f"\nCODEBOOK DOCUMENTS (for PROJECT {project_id}):\n"
        for i, chunk in enumerate(codebook_chunks, 1):
            prompt += f"\n[CODEBOOK-{i}] {chunk.discipline.upper()} | {chunk.source_doc} p.{chunk.page}\n{chunk.text}\n"
    
    if spec_chunks:
        prompt += f"\nSPECIFICATION DOCUMENTS (for PROJECT {project_id}):\n"
        for i, chunk in enumerate(spec_chunks, 1):
            prompt += f"\n[SPEC-{i}] {chunk.discipline.upper()} | {chunk.source_doc} p.{chunk.page}\n{chunk.text}\n"
    
    if is_compliance_question(query):
        prompt += f"""

INSTRUCTIONS:
1. Analyze the query against ALL provided documents for the SAME PROJECT ({project_id})
2. The Codebook and Specification documents are from the SAME PROJECT - do not treat them as separate projects
3. If documents contain conflicting requirements, identify the conflict and recommend resolution
4. Provide a clear, technical answer based on the complete document set for this single project
5. Structure compliance responses as:

Status: [PASS/FAIL/WARNING/UNKNOWN]
Severity: [HIGH/MEDIUM/LOW/N/A] 
Required: [What the regulation specifies]
Actual: [What the project shows, if available]

[Then provide detailed explanation including any conflicts between documents]

IMPORTANT: Remember that all documents relate to PROJECT {project_id}. Focus on factual analysis."""
    else:
        prompt += f"""

INSTRUCTIONS:
Provide a clear, technical answer based on ALL the provided documents for PROJECT {project_id}. 
The Codebook and Specification documents are from the SAME PROJECT - do not treat them as separate projects.
If documents contain different information on the same topic, explain the differences and their context. 
Focus on factual information from the complete document set for this single project."""
    
    return prompt


def generate_answer(
    query: str, 
    context_chunks: List[Chunk], 
    api_key: Optional[str] = None
) -> LLMResult:
    """
    Generate answer using Groq LLM with the provided context.
    
    Args:
        query: User's question
        context_chunks: Retrieved document chunks for context
        api_key: Groq API key (optional - falls back to local response if missing)
    
    Returns:
        LLMResult with answer and optional compliance check
    """
    
    prompt = _build_prompt(query, context_chunks)
    
    # If no API key provided, return a helpful fallback response
    if not api_key:
        fallback_answer = f"""I can see {len(context_chunks)} relevant document chunks for your query: "{query}"

However, I need a Groq API key to provide an AI-generated answer. You can get a free API key at console.groq.com.

The retrieved documents contain information about:
{', '.join(set(c.discipline for c in context_chunks))}

From sources: {', '.join(set(c.source_doc for c in context_chunks))}

Please add your Groq API key in the sidebar to get detailed AI analysis of these documents."""
        
        return LLMResult(
            answer=fallback_answer,
            compliance_check=None,
            prompt=prompt
        )
    
    # Call Groq API
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated working Groq model
            messages=[
                {
                    "role": "system", 
                    "content": "You are a construction compliance expert. Analyze building codes and project specifications to provide accurate, technical answers."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.1,  # Low temperature for factual responses
        )
        
        response_text = completion.choices[0].message.content
        
        # Extract compliance info if it's a compliance question
        compliance_check = None
        if is_compliance_question(query):
            compliance_check = _extract_compliance_info(response_text)
        
        return LLMResult(
            answer=response_text,
            compliance_check=compliance_check,
            prompt=prompt
        )
        
    except Exception as e:
        error_answer = f"""Error calling Groq API: {str(e)}

Please check:
1. Your API key is correct (get one free at console.groq.com)
2. You have internet connectivity
3. Your Groq account has available credits

Retrieved {len(context_chunks)} relevant chunks from your documents, but cannot generate AI response without valid API access."""
        
        return LLMResult(
            answer=error_answer,
            compliance_check=None,
            prompt=prompt
        )