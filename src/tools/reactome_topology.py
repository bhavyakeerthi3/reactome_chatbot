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

    def get_flow_context(self, st_id: str) -> str:
        """Get a human-readable summary of the topological flow for an event using a single API call."""
        data = self.query_id(st_id)
        if not data:
            return ""

        context = f"Event: {data.get('displayName', st_id)} ({st_id})\n"
        
        # Extract inputs/outputs/catalysts (for Reactions)
        inputs = [i.get("displayName") for i in data.get("input", [])]
        outputs = [o.get("displayName") for o in data.get("output", [])]
        catalysts = [c.get("physicalEntity", {}).get("displayName") for c in data.get("catalystActivity", []) if c.get("physicalEntity")]
        
        # Extract preceding events
        preceding = [e.get("displayName") for e in data.get("precedingEvent", [])]

        if inputs:
            context += f"Inputs: {', '.join(inputs)}\n"
        if outputs:
            context += f"Outputs: {', '.join(outputs)}\n"
        if catalysts:
            context += f"Catalysts: {', '.join(catalysts)}\n"
        if preceding:
            context += f"Preceding Events: {', '.join(preceding)}\n"
            
        return context
