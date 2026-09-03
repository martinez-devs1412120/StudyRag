"""Services layer: business logic that routers call.

Keep services framework-agnostic (no FastAPI imports) so they can be reused
by background jobs, scripts, or tests without spinning up a server.
"""
