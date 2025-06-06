import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chatbot_monitoring')

class Monitoring:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Métriques
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "category_distribution": {},
            "error_types": {}
        }
        
        # Initialisation des fichiers de log
        self.performance_log = self.log_dir / "performance.log"
        self.error_log = self.log_dir / "errors.log"
        self.usage_log = self.log_dir / "usage.log"
        
        # Création des fichiers s'ils n'existent pas
        for log_file in [self.performance_log, self.error_log, self.usage_log]:
            if not log_file.exists():
                log_file.touch()

    def log_request(self, 
                   user_input: str, 
                   response: Dict[str, Any], 
                   processing_time: float,
                   error: Optional[Exception] = None) -> None:
        """
        Enregistre les détails d'une requête
        """
        timestamp = datetime.now().isoformat()
        
        # Mise à jour des métriques
        self.metrics["total_requests"] += 1
        if error:
            self.metrics["failed_requests"] += 1
            error_type = type(error).__name__
            self.metrics["error_types"][error_type] = self.metrics["error_types"].get(error_type, 0) + 1
        else:
            self.metrics["successful_requests"] += 1
            
        # Mise à jour du temps de réponse moyen
        self.metrics["total_response_time"] += processing_time
        self.metrics["average_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["total_requests"]
        )
        
        # Mise à jour de la distribution des catégories
        if response.get("category"):
            category = response["category"]
            self.metrics["category_distribution"][category] = (
                self.metrics["category_distribution"].get(category, 0) + 1
            )
        
        # Log de performance
        performance_data = {
            "timestamp": timestamp,
            "processing_time": processing_time,
            "input_length": len(user_input),
            "response_length": len(response.get("text", "")),
            "category": response.get("category"),
            "confidence": response.get("confidence")
        }
        
        with open(self.performance_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(performance_data) + "\n")
        
        # Log d'erreur si nécessaire
        if error:
            error_data = {
                "timestamp": timestamp,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "user_input": user_input
            }
            with open(self.error_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_data) + "\n")
        
        # Log d'utilisation
        usage_data = {
            "timestamp": timestamp,
            "user_input": user_input,
            "response": response,
            "processing_time": processing_time
        }
        with open(self.usage_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(usage_data) + "\n")
        
        # Log dans la console
        logger.info(
            f"Request processed in {processing_time:.2f}s - "
            f"Category: {response.get('category', 'N/A')} - "
            f"Confidence: {response.get('confidence', 'N/A'):.2f}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques actuelles
        """
        return self.metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des performances
        """
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            ),
            "average_response_time": self.metrics["average_response_time"],
            "category_distribution": self.metrics["category_distribution"],
            "error_types": self.metrics["error_types"]
        }

    def reset_metrics(self) -> None:
        """
        Réinitialise les métriques
        """
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "category_distribution": {},
            "error_types": {}
        } 