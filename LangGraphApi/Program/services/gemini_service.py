# /services/gemini_service.py
import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica le variabili d'ambiente dal file .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configura l'SDK di Google
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configurata correttamente")
else:
    raise ValueError("GEMINI_API_KEY non trovata. Assicurati di aver creato un file .env e di aver impostato la chiave API.")

def call_gemini_api(prompt: str, base64_data: str, mime_type: str) -> dict:
    """
    Chiama l'API di Gemini utilizzando l'SDK ufficiale di Google.

    Args:
        prompt: Il prompt testuale da inviare al modello.
        base64_data: I dati del file codificati in base64.
        mime_type: Il tipo MIME del file.

    Returns:
        Un dizionario contenente la risposta parsata o un messaggio di errore.
        NOTA: Può restituire anche una lista se Gemini restituisce direttamente un array JSON.
    """
    model_name = "gemini-2.5-pro"  # Usa il modello più recente disponibile
    
    logger.info(f"Chiamata a Gemini con modello: {model_name}")
    logger.info(f"Lunghezza prompt: {len(prompt)} caratteri")
    logger.info(f"Lunghezza dati base64: {len(base64_data)} caratteri")
    logger.info(f"MIME type: {mime_type}")
    
    # Crea il modello generativo
    model = genai.GenerativeModel(model_name)

    # Prepara il contenuto multimodale
    content_parts = [
        prompt,
        {"mime_type": mime_type, "data": base64_data}
    ]

    # Configurazione per ottenere una risposta JSON
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.1,  # Temperatura bassa per risultati più consistenti
        top_p=0.95,
        top_k=40
    )

    try:
        logger.info("Invio richiesta a Gemini API...")
        
        # Invia la richiesta all'API
        response = model.generate_content(
            contents=content_parts,
            generation_config=generation_config
        )
        
        logger.info("Risposta ricevuta da Gemini API")
        
        # Verifica che la risposta contenga testo
        if not response.text:
            logger.warning("La risposta di Gemini è vuota")
            return {"error": "Risposta vuota da Gemini API"}
        
        logger.info(f"Risposta Gemini (primi 500 caratteri): {response.text[:500]}...")
        
        # Estrae e parsa il contenuto JSON dalla risposta
        try:
            parsed_response = json.loads(response.text)
            
            # Log del tipo di risposta
            if isinstance(parsed_response, dict):
                logger.info(f"JSON parsato come dizionario. Chiavi: {list(parsed_response.keys())}")
            elif isinstance(parsed_response, list):
                logger.info(f"JSON parsato come lista con {len(parsed_response)} elementi")
                if len(parsed_response) > 0:
                    logger.info(f"Primo elemento: {parsed_response[0] if len(str(parsed_response[0])) < 200 else str(parsed_response[0])[:200] + '...'}")
            else:
                logger.info(f"JSON parsato come tipo: {type(parsed_response)}")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"Errore nel parsing JSON: {e}")
            logger.error(f"Testo ricevuto: {response.text}")
            
            # Prova a pulire il testo e riparsare
            cleaned_text = response.text.strip()
            
            # Rimuovi eventuali backtick markdown
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            try:
                parsed_response = json.loads(cleaned_text)
                logger.info("JSON parsato dopo pulizia del testo")
                return parsed_response
            except json.JSONDecodeError:
                return {"error": f"Errore nel parsing della risposta JSON: {e}"}

    except Exception as e:
        # Gestisce eventuali errori durante la chiamata API
        error_msg = f"Errore durante la chiamata a Gemini: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Prova a fornire più dettagli sull'errore
        if hasattr(e, 'message'):
            error_msg += f" - Dettagli: {e.message}"
        
        return {"error": error_msg}