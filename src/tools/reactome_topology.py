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

    def query_id(self, st_id: str) -> Optional[dict[str, Any]]:
        """Queries the Content Service for a specific Stable ID."""
        url = f"{self.BASE_URL}/query/{st_id}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error querying Reactome ID {st_id}: {e}")
            return None

    def get_reaction_participants(self, st_id: str) -> dict[str, list[str]]:
        """
        Fetches the inputs, outputs, and catalysts for a given reaction.
        """
        data = self.query_id(st_id)
        if not data:
            return {}

        participants = {
            "inputs": [i.get("displayName") for i in data.get("input", [])],
            "outputs": [o.get("displayName") for o in data.get("output", [])],
            "catalysts": [
                c.get("physicalEntity", {}).get("displayName")
                for c in data.get("catalystActivity", [])
            ],
        }
        return participants

    def get_preceding_events(self, st_id: str) -> list[dict[str, str]]:
        """
        Fetches events that immediately precede the given event.
        """
        data = self.query_id(st_id)
        if not data:
            return []

        preceding = [
            {"stId": e.get("stId"), "displayName": e.get("displayName")}
            for e in data.get("precedingEvent", [])
        ]
        return preceding

    def get_flow_context(self, st_id: str) -> str:
        """
        Generates a human-readable summary of the topological flow for an event.
        """
        participants = self.get_reaction_participants(st_id)
        preceding = self.get_preceding_events(st_id)
        
        data = self.query_id(st_id)
        name = data.get("displayName") if data else st_id

        summary = f"Reaction: {name} ({st_id})\n"
        if participants.get("inputs"):
            summary += f"- Inputs: {', '.join(participants['inputs'])}\n"
        if participants.get("outputs"):
            summary += f"- Outputs: {', '.join(participants['outputs'])}\n"
        if participants.get("catalysts"):
            summary += f"- Catalysts: {', '.join(participants['catalysts'])}\n"
        
        if preceding:
            summary += "- Preceded by:\n"
            for p in preceding:
                summary += f"  * {p['displayName']} ({p['stId']})\n"
        
        return summary
