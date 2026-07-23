"""Web bridge: play the cabt game in a browser against a bot.

The React UI in ``replay/07_vite_react_cards`` renders the board and turns the
engine's ``select`` prompts into clickable buttons; ``pkm.web.server`` is the
thin HTTP layer that connects it to the same ``ThreadedEnvSession`` the Textual
TUI uses. See ``server.py`` for the protocol.
"""
