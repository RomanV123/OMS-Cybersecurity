import requests
import jwt


token = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjkwNDIwOTU2NSIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL25hbWUiOiI8dXNlcm5hbWU-IiwiZXhwIjozNzg2OTEyMDAwfQ."

unverified_payload = jwt.decode(token, options = {"verify_signature": False})
print(unverified_payload)

