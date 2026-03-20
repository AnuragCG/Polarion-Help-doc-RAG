class SessionMemory:
    def __init__(self):
        self.sessions = {}

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def add_interaction(self, session_id, query, answer):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "query": query,
            "answer": answer
        })

    def get_last_interactions(self, session_id, k=3):
        return self.sessions.get(session_id, [])[-k:]