import requests
from typing import Any, Optional
from util.logging import logging

class ReactomeTopologyTool:
    """
    A tool to query the Reactome Content Service for topological information
    about pathways and reactions (e.g., inputs, outputs, preceding/subsequent events).
    """

    BASE_URL = "https://reactome.org/ContentService/data"

    def __init__(self):
        self.session = requests.Session()

    def query_id(self, st_id: str) -> dict[str, Any] | None:
        """Query the Content Service for a single ID."""
        url = f"{self.BASE_URL}/query/{st_id}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def get_reaction_participants(self, st_id: str) -> dict[str, Any]:
        """Fetch inputs, outputs, and catalysts for a given reaction."""
        url = f"{self.BASE_URL}/participants/{st_id}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "inputs": [p.get("displayName") for p in data.get("inputs", [])],
                    "outputs": [p.get("displayName") for p in data.get("outputs", [])],
                    "catalysts": [c.get("displayName") for c in data.get("catalysts", [])],
                }
        except Exception:
            pass
        return {}

    def get_preceding_events(self, st_id: str) -> list[str]:
        """Fetch preceding events for a given reaction or event."""
        url = f"{self.BASE_URL}/precedingEvents/{st_id}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [e.get("displayName") for e in data]
        except Exception:
            pass
        return []

    def get_flow_context(self, st_id: str) -> str:
        """Get a human-readable summary of the topological flow for an event."""
        participants = self.get_reaction_participants(st_id)
        preceding = self.get_preceding_events(st_id)
        
        context = f"Event: {st_id}\n"
        if participants.get("inputs"):
            context += f"Inputs: {', '.join(participants['inputs'])}\n"
        if participants.get("outputs"):
            context += f"Outputs: {', '.join(participants['outputs'])}\n"
        if preceding:
            context += f"Preceding Events: {', '.join(preceding)}\n"
            
        return context if context != f"Event: {st_id}\n" else ""
